import os
from typing import Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
import torchvision
import wandb
from torch import Tensor, nn
from tqdm import tqdm

from config_schema import MainConfig
from geo_text import expand_prompts, load_geo_vocab
from surrogates.gsdm_generator import GSDMGenerator
from utils import (
    ImageFolderWithPaths,
    build_image_transform,
    ensure_dir,
    get_models,
    hash_training_config,
    set_environment,
    setup_wandb,
)


def average_template_features(features: Tensor, num_labels: int, num_templates: int) -> Tensor:
    features = features.view(num_labels, num_templates, -1).mean(dim=1)
    return F.normalize(features, dim=1)


def total_variation(delta: Tensor) -> Tensor:
    vertical = torch.mean(torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]))
    return vertical + horizontal


def _mask_4d(mask: Tensor, like: Optional[Tensor] = None) -> Tensor:
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    elif mask.dim() != 4 or mask.size(1) != 1:
        raise ValueError(f"Expected mask shape [B, H, W] or [B, 1, H, W], got {tuple(mask.shape)}")
    if like is not None:
        mask = mask.to(device=like.device, dtype=like.dtype)
    return mask.clamp(0.0, 1.0)


def lowpass_fft(tensor: Tensor, ratio: float) -> Tensor:
    """Keep only the centered low-frequency FFT window of a BCHW tensor."""
    ratio = float(ratio)
    if ratio <= 0.0 or ratio >= 1.0:
        return tensor

    _, _, height, width = tensor.shape
    keep_h = max(1, int(round(height * ratio)))
    keep_w = max(1, int(round(width * ratio)))
    y0 = max(0, height // 2 - keep_h // 2)
    x0 = max(0, width // 2 - keep_w // 2)
    y1 = min(height, y0 + keep_h)
    x1 = min(width, x0 + keep_w)

    fft = torch.fft.fftshift(torch.fft.fft2(tensor, dim=(-2, -1)), dim=(-2, -1))
    mask = torch.zeros_like(fft)
    mask[:, :, y0:y1, x0:x1] = 1.0
    filtered = torch.fft.ifft2(
        torch.fft.ifftshift(fft * mask, dim=(-2, -1)),
        dim=(-2, -1),
    ).real
    return filtered


def mix_background_lowfreq_grad(cfg, grad: Tensor, mask: Tensor) -> Tensor:
    if not bool(cfg.hge.bg_lowfreq_enabled):
        return grad
    mask = _mask_4d(mask, grad).expand_as(grad)
    bg_mask = 1.0 - mask
    if torch.max(bg_mask) <= 0:
        return grad

    bg_grad = lowpass_fft(grad * bg_mask, float(cfg.hge.bg_lowfreq_ratio))
    return grad * mask + bg_grad * bg_mask


def dilate_mask(mask: Tensor, kernel_size: int) -> Tensor:
    """Dilate a [B, H, W] or [B, 1, H, W] mask with max pooling."""
    mask_4d = _mask_4d(mask)

    if kernel_size <= 1:
        dilated = mask_4d
    else:
        if kernel_size % 2 == 0:
            kernel_size += 1
        dilated = F.max_pool2d(mask_4d, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return dilated.squeeze(1).clamp(0.0, 1.0)


def build_context_crop(
    image: Tensor,
    mask: Tensor,
    input_res: int,
    padding_ratio: float,
) -> Tensor:
    """Crop each image to the dilated mask bbox plus context, then resize."""
    if mask.dim() == 4:
        mask = mask.squeeze(1)

    crops = []
    _, _, image_h, image_w = image.shape
    for batch_index in range(image.size(0)):
        active = torch.nonzero(mask[batch_index] > 0.5, as_tuple=False)
        if active.numel() == 0:
            crop = image[batch_index : batch_index + 1]
        else:
            y_min = int(active[:, 0].min().item())
            y_max = int(active[:, 0].max().item())
            x_min = int(active[:, 1].min().item())
            x_max = int(active[:, 1].max().item())

            bbox_h = y_max - y_min + 1
            bbox_w = x_max - x_min + 1
            pad_y = int(round(bbox_h * padding_ratio))
            pad_x = int(round(bbox_w * padding_ratio))

            y0 = max(0, y_min - pad_y)
            y1 = min(image_h, y_max + pad_y + 1)
            x0 = max(0, x_min - pad_x)
            x1 = min(image_w, x_max + pad_x + 1)
            crop = image[batch_index : batch_index + 1, :, y0:y1, x0:x1]

        crops.append(
            F.interpolate(
                crop,
                size=(input_res, input_res),
                mode="bilinear",
                align_corners=False,
            )
        )
    return torch.cat(crops, dim=0)


def build_protected_eps_map(
    cfg,
    mask: Tensor,
    image: Tensor,
    fallback_flags: List[bool],
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    mask_4d = _mask_4d(mask, image)
    area = (mask_4d.squeeze(1) > 0.5).float().mean(dim=(1, 2))
    fallback = torch.tensor(fallback_flags, dtype=torch.bool, device=image.device)
    cap_mask = fallback | (area >= float(cfg.hge.high_epsilon_max_mask_area))

    base_eps = torch.full_like(area, float(cfg.hge.mask_epsilon), dtype=image.dtype)
    cap_eps = torch.full_like(area, float(cfg.hge.fallback_mask_epsilon), dtype=image.dtype)
    mask_eps = torch.where(cap_mask, torch.minimum(base_eps, cap_eps), base_eps).view(-1, 1, 1, 1)

    mask_expanded = mask_4d.expand_as(image)
    eps_map = mask_expanded * mask_eps + (1.0 - mask_expanded) * float(cfg.hge.bg_epsilon)
    logs = {
        "mask_area": area.mean().item(),
        "mask_area_min": area.min().item(),
        "mask_area_max": area.max().item(),
        "gsdm_fallback": float(any(fallback_flags)),
        "effective_mask_epsilon": mask_eps.float().mean().item(),
    }
    return eps_map, mask_eps.squeeze(), logs


def delta_region_stats(delta: Tensor, mask: Tensor) -> Dict[str, float]:
    mask_region = _mask_4d(mask, delta).expand_as(delta) > 0.5
    bg_region = ~mask_region
    abs_delta = torch.abs(delta.detach())

    def _region_max(region: Tensor) -> float:
        if not torch.any(region):
            return 0.0
        return abs_delta[region].max().item()

    def _region_mean(region: Tensor) -> float:
        if not torch.any(region):
            return 0.0
        return abs_delta[region].mean().item()

    return {
        "max_delta_mask": _region_max(mask_region),
        "max_delta_bg": _region_max(bg_region),
        "mean_delta_mask": _region_mean(mask_region),
        "mean_delta_bg": _region_mean(bg_region),
    }


class GeoTextBank:
    def __init__(self, extractors: List[nn.Module], cfg):
        self.extractors = nn.ModuleList(extractors)
        self.vocab = load_geo_vocab(cfg.hge.vocab_path)
        self.features = self._build_feature_cache()

    @torch.no_grad()
    def _build_feature_cache(self) -> Dict[int, Dict[str, Tensor]]:
        cache = {}
        for model_index, extractor in enumerate(self.extractors):
            country_prompts = expand_prompts(self.vocab.countries, self.vocab.country_templates)
            city_prompts = expand_prompts(self.vocab.cities, self.vocab.city_templates)
            country_evidence_prompts = expand_prompts(
                self.vocab.countries,
                self.vocab.country_evidence_templates,
            )
            city_evidence_prompts = expand_prompts(
                self.vocab.cities,
                self.vocab.city_evidence_templates,
            )

            country_features = extractor.encode_texts(country_prompts)
            city_features = extractor.encode_texts(city_prompts)
            country_evidence_features = extractor.encode_texts(country_evidence_prompts)
            city_evidence_features = extractor.encode_texts(city_evidence_prompts)

            cache[model_index] = {
                "country": average_template_features(
                    country_features,
                    num_labels=len(self.vocab.countries),
                    num_templates=len(self.vocab.country_templates),
                ),
                "city": average_template_features(
                    city_features,
                    num_labels=len(self.vocab.cities),
                    num_templates=len(self.vocab.city_templates),
                ),
                "country_evidence": average_template_features(
                    country_evidence_features,
                    num_labels=len(self.vocab.countries),
                    num_templates=len(self.vocab.country_evidence_templates),
                ),
                "city_evidence": average_template_features(
                    city_evidence_features,
                    num_labels=len(self.vocab.cities),
                    num_templates=len(self.vocab.city_evidence_templates),
                ),
            }
        return cache


def _feature_items(feature_dict):
    if isinstance(feature_dict, dict):
        return feature_dict.items()
    return [(0, feature_dict)]


@torch.no_grad()
def capture_topk_anchors(
    feature_dict,
    geo_bank: GeoTextBank,
    k_country: int,
    k_city: int,
) -> Tuple[Dict[int, Tensor], Dict[int, Tensor]]:
    country_topk = {}
    city_topk = {}
    for model_index, image_features in _feature_items(feature_dict):
        country_features = geo_bank.features[model_index]["country"].to(image_features.device)
        city_features = geo_bank.features[model_index]["city"].to(image_features.device)

        country_logits = image_features @ country_features.T
        city_logits = image_features @ city_features.T

        country_topk[model_index] = torch.topk(
            country_logits,
            k=min(k_country, country_logits.size(-1)),
            dim=-1,
        ).indices
        city_topk[model_index] = torch.topk(
            city_logits,
            k=min(k_city, city_logits.size(-1)),
            dim=-1,
        ).indices
    return country_topk, city_topk


def _gather_anchor_features(text_features: Tensor, topk_idx: Tensor) -> Tensor:
    batch_size, topk = topk_idx.shape
    dim = text_features.size(-1)
    expanded = text_features.unsqueeze(0).expand(batch_size, -1, -1)
    gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(expanded, dim=1, index=gather_idx)


def entropy_stats(logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-8)
    negative_entropy = torch.sum(probs * log_probs, dim=-1).mean()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    top1_prob = probs.max(dim=-1).values.mean()
    return negative_entropy, entropy, top1_prob


def topk_cosine_loss(feature_dict, geo_bank: GeoTextBank, topk_dict, level: str) -> Tensor:
    losses = []
    for model_index, image_features in _feature_items(feature_dict):
        text_features = geo_bank.features[model_index][level].to(image_features.device)
        anchor_features = _gather_anchor_features(
            text_features,
            topk_dict[model_index].to(image_features.device),
        )
        cosine_logits = torch.bmm(
            anchor_features,
            image_features.unsqueeze(-1),
        ).squeeze(-1)
        losses.append(cosine_logits.mean())
    return torch.stack(losses).mean()


class HierarchicalGeoEntropyLoss(nn.Module):
    def __init__(self, geo_bank: GeoTextBank, cfg):
        super().__init__()
        self.geo_bank = geo_bank
        self.lambda_country = float(cfg.hge.lambda_country)
        self.lambda_city = float(cfg.hge.lambda_city)
        self.temperature = float(cfg.hge.temperature)
        self.topk_suppress_weight = float(cfg.hge.topk_suppress_weight)

    def forward(self, feature_dict, country_topk, city_topk, level: str):
        if level not in {"country", "city"}:
            raise ValueError(f"level must be 'country' or 'city', got {level!r}")

        losses = []
        logs = {
            f"loss_{level}": [],
            f"entropy_loss_{level}": [],
            f"suppress_loss_{level}": [],
            f"topk_cosine_{level}": [],
            f"entropy_{level}": [],
            f"top1_{level}_prob": [],
        }

        for model_index, image_features in _feature_items(feature_dict):
            text_features = self.geo_bank.features[model_index][level].to(image_features.device)
            topk_source = country_topk if level == "country" else city_topk
            anchor_features = _gather_anchor_features(
                text_features,
                topk_source[model_index].to(image_features.device),
            )

            cosine_logits = torch.bmm(
                anchor_features,
                image_features.unsqueeze(-1),
            ).squeeze(-1)
            logits = cosine_logits / self.temperature

            entropy_loss, entropy_value, top1_prob = entropy_stats(logits)
            suppress_loss = cosine_logits.mean()
            loss_value = entropy_loss + self.topk_suppress_weight * suppress_loss
            losses.append(loss_value)

            logs[f"loss_{level}"].append(loss_value.detach())
            logs[f"entropy_loss_{level}"].append(entropy_loss.detach())
            logs[f"suppress_loss_{level}"].append(suppress_loss.detach())
            logs[f"topk_cosine_{level}"].append(suppress_loss.detach())
            logs[f"entropy_{level}"].append(entropy_value.detach())
            logs[f"top1_{level}_prob"].append(top1_prob.detach())

        scalar_logs = {
            key: torch.stack(values).mean().item()
            for key, values in logs.items()
        }
        return torch.stack(losses).mean(), scalar_logs

    def suppress_only(self, feature_dict, country_topk, city_topk, level: str) -> Tensor:
        if level not in {"country", "city"}:
            raise ValueError(f"level must be 'country' or 'city', got {level!r}")
        topk_source = country_topk if level == "country" else city_topk
        return topk_cosine_loss(feature_dict, self.geo_bank, topk_source, level)


def log_metrics(pbar, metrics, img_index, epoch=None):
    pbar.set_postfix({
        key: f"{value:.5f}" if "prob" in key or "loss" in key else f"{value:.3f}"
        for key, value in metrics.items()
    })
    if hasattr(wandb, "log"):
        wandb_metrics = {f"img{img_index}_{key}": value for key, value in metrics.items()}
        if epoch is not None:
            wandb_metrics["epoch"] = epoch
        wandb.log(wandb_metrics)


def generate_geo_masks(cfg, image_org, gsdm_generator):
    masks = []
    fallback_flags = []
    for batch_index in range(image_org.size(0)):
        result = gsdm_generator.generate_mask(
            image_org[batch_index],
            cfg.gsdm.geo_label,
        )
        if isinstance(result, tuple):
            mask = result[0]
        else:
            mask = result
        fallback = bool((mask > 0.5).float().mean().item() >= 0.999)
        if fallback:
            print(
                f"  [GSDM] Image {batch_index}: no geo-features detected "
                "-> uniform mask (HGE attack)"
            )
        masks.append(mask)
        fallback_flags.append(fallback)
    return torch.stack(masks, dim=0).to(cfg.model.device), fallback_flags


def hge_attack(
    cfg,
    ensemble_extractor,
    hge_loss,
    img_index,
    image_org,
    gsdm_generator,
):
    mask_tensor, fallback_flags = generate_geo_masks(cfg, image_org, gsdm_generator)
    context_mask = dilate_mask(mask_tensor, int(cfg.hge.context_dilation_kernel))
    eps_map, _, mask_logs = build_protected_eps_map(
        cfg,
        mask_tensor,
        image_org,
        fallback_flags,
    )

    with torch.no_grad():
        original_features = ensemble_extractor(image_org)
        country_topk, city_topk = capture_topk_anchors(
            original_features,
            hge_loss.geo_bank,
            k_country=cfg.hge.k_country,
            k_city=cfg.hge.k_city,
        )

    delta = torch.zeros_like(image_org, requires_grad=True)
    pbar = tqdm(range(cfg.optim.steps), desc="Hierarchical Dynamic Top-K HGE")

    for epoch in pbar:
        adv_image = image_org + delta
        global_features = ensemble_extractor(adv_image)
        country_loss_global, country_logs = hge_loss(
            global_features,
            country_topk,
            city_topk,
            level="country",
        )
        city_suppress_global = hge_loss.suppress_only(
            global_features,
            country_topk,
            city_topk,
            level="city",
        )

        context_image = build_context_crop(
            adv_image,
            context_mask,
            input_res=cfg.model.input_res,
            padding_ratio=float(cfg.hge.context_padding_ratio),
        )
        context_features = ensemble_extractor(context_image)
        city_loss_context, city_logs = hge_loss(
            context_features,
            country_topk,
            city_topk,
            level="city",
        )

        loss_hge = (
            cfg.hge.lambda_country * country_loss_global
            + cfg.hge.lambda_city * city_loss_context
            + cfg.hge.lambda_city_global_suppress * city_suppress_global
        )
        loss_tv = total_variation(delta)
        loss = cfg.hge.lambda_hge * loss_hge + cfg.hge.tv_weight * loss_tv

        metrics = {
            **delta_region_stats(delta, mask_tensor),
            **mask_logs,
            "bg_lowfreq_enabled": float(bool(cfg.hge.bg_lowfreq_enabled)),
            "bg_lowfreq_ratio": float(cfg.hge.bg_lowfreq_ratio),
            "loss_country_global": country_loss_global.item(),
            "loss_city_context": city_loss_context.item(),
            "loss_city_global_suppress": city_suppress_global.item(),
            "entropy_loss_country_global": country_logs["entropy_loss_country"],
            "entropy_loss_city_context": city_logs["entropy_loss_city"],
            "suppress_loss_country_global": country_logs["suppress_loss_country"],
            "suppress_loss_city_context": city_logs["suppress_loss_city"],
            "topk_cosine_country_global": country_logs["topk_cosine_country"],
            "topk_cosine_city_context": city_logs["topk_cosine_city"],
            "entropy_country_global": country_logs["entropy_country"],
            "entropy_city_context": city_logs["entropy_city"],
            "top1_country_prob_global": country_logs["top1_country_prob"],
            "top1_city_prob_context": city_logs["top1_city_prob"],
            "hge_loss": loss_hge.item(),
            "tv_loss": loss_tv.item(),
            "total_loss": loss.item(),
        }
        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
        grad = mix_background_lowfreq_grad(cfg, grad, mask_tensor)
        with torch.no_grad():
            delta_next = delta - cfg.optim.alpha * torch.sign(grad)
            delta.copy_(torch.max(torch.min(delta_next, eps_map), -eps_map))
        delta.grad = None

    adv_image = torch.clamp((image_org + delta) / 255.0, 0.0, 1.0)
    log_metrics(
        pbar,
        delta_region_stats(delta, mask_tensor),
        img_index,
    )
    return adv_image


def save_outputs(cfg, config_hash, adv_image, path_org):
    for path_idx in range(len(path_org)):
        folder = os.path.basename(os.path.dirname(path_org[path_idx]))
        name = os.path.basename(path_org[path_idx])
        folder_to_save = os.path.join(cfg.data.output, "img", config_hash, folder)
        ensure_dir(folder_to_save)

        ext = os.path.splitext(name)[1].lower()
        save_name = os.path.splitext(name)[0] + ".png" if ext in [".jpg", ".jpeg"] else name
        torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, save_name))


@hydra.main(version_base=None, config_path="config", config_name="hge")
def main(cfg: MainConfig):
    set_environment()
    if str(cfg.attack).lower() != "hge":
        raise ValueError("hge.py only runs HGE geoprivacy attack. Use attack=hge.")

    setup_wandb(cfg, tags=["hge", "geoprivacy", "dynamic_topk"])
    if hasattr(wandb, "define_metric"):
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    ensemble_extractor, models = get_models(cfg)
    geo_bank = GeoTextBank(models, cfg)
    hge_loss = HierarchicalGeoEntropyLoss(geo_bank, cfg)

    transform_fn = build_image_transform(cfg)
    source_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    source_loader = torch.utils.data.DataLoader(
        source_data, batch_size=cfg.data.batch_size, shuffle=False
    )

    gsdm_generator = GSDMGenerator(
        device=cfg.model.device,
        box_threshold=cfg.gsdm.box_threshold,
        text_threshold=cfg.gsdm.text_threshold,
        mask_fusion=cfg.gsdm.mask_fusion,
    )
    config_hash = hash_training_config(cfg)

    total = max(1, cfg.data.num_samples // cfg.data.batch_size)
    for img_index, (image_org, _, path_org) in enumerate(source_loader):
        if cfg.data.batch_size * (img_index + 1) > cfg.data.num_samples:
            break
        print(f"\nProcessing HGE image {img_index + 1}/{total}")
        image_org = image_org.to(cfg.model.device)
        adv_image = hge_attack(
            cfg,
            ensemble_extractor,
            hge_loss,
            img_index,
            image_org,
            gsdm_generator,
        )
        save_outputs(cfg, config_hash, adv_image, path_org)

    if hasattr(wandb, "finish"):
        wandb.finish()


if __name__ == "__main__":
    main()

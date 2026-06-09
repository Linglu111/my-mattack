import os

import hydra
import torch
import torchvision
import wandb
from tqdm import tqdm

from config_schema import MainConfig
from surrogates.gsdm_generator import GSDMGenerator
from utils import (
    ImageFolderWithPaths,
    build_image_transform,
    build_source_crop,
    build_target_crop,
    ensure_dir,
    get_ensemble_loss,
    get_models,
    hash_training_config,
    set_environment,
    setup_wandb,
)


def log_metrics(pbar, metrics, img_index, epoch=None):
    pbar.set_postfix({
        key: f"{value:.5f}" if "sim" in key else f"{value:.3f}"
        for key, value in metrics.items()
    })
    if hasattr(wandb, "log"):
        wandb_metrics = {f"img{img_index}_{key}": value for key, value in metrics.items()}
        if epoch is not None:
            wandb_metrics["epoch"] = epoch
        wandb.log(wandb_metrics)


def generate_geo_masks(cfg, image_org, gsdm_generator):
    masks = []
    for batch_index in range(image_org.size(0)):
        result = gsdm_generator.generate_mask(
            image_org[batch_index],
            cfg.gsdm.geo_label,
        )
        if isinstance(result, tuple):
            mask = result[0]
            if len(result) > 1 and result[1].get("fallback"):
                print(
                    f"  [GSDM] Image {batch_index}: no geo-features detected "
                    "-> uniform mask (target attack)"
                )
        else:
            mask = result
        masks.append(mask)
    return torch.stack(masks, dim=0).to(cfg.model.device)


def target_attack(
    cfg,
    ensemble_extractor,
    ensemble_loss,
    source_crop,
    target_crop,
    img_index,
    image_org,
    image_tgt,
    gsdm_generator,
):
    mask_tensor = generate_geo_masks(cfg, image_org, gsdm_generator)
    mask_expanded = mask_tensor.unsqueeze(1).expand_as(image_org)
    delta = torch.zeros_like(image_org, requires_grad=True)

    lpips_model = None
    if cfg.gsdm.use_lpips:
        try:
            import lpips

            lpips_model = lpips.LPIPS(net="alex").to(cfg.model.device)
            lpips_model.eval()
        except ImportError:
            print("Warning: lpips not installed, skipping perceptual loss")

    pbar = tqdm(range(cfg.optim.steps), desc="M-Attack progress")
    for epoch in pbar:
        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        adv_image = image_org + delta
        adv_features = ensemble_extractor(adv_image)

        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        global_sim = ensemble_loss(adv_features)
        metrics["global_similarity"] = global_sim.item()
        loss = -global_sim

        if lpips_model is not None:
            adv_normalized = torch.clamp(adv_image / 255.0, 0.0, 1.0)
            org_normalized = torch.clamp(image_org / 255.0, 0.0, 1.0)
            loss_lpips = lpips_model(adv_normalized, org_normalized).mean()
            loss = loss + cfg.gsdm.lpips_weight * loss_lpips
            metrics["lpips_loss"] = loss_lpips.item()

        if cfg.model.use_source_crop:
            local_features = ensemble_extractor(source_crop(adv_image))
            local_sim = ensemble_loss(local_features)
            loss = loss - local_sim
            metrics["local_similarity"] = local_sim.item()

        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad) * mask_expanded,
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )
        delta.grad = None

    adv_image = torch.clamp((image_org + delta) / 255.0, 0.0, 1.0)
    log_metrics(
        pbar,
        {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        },
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


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    set_environment()
    if str(cfg.attack).lower() != "gsdm":
        raise ValueError("mattack.py only runs the original target attack. Use attack=gsdm.")

    setup_wandb(cfg, tags=["mattack", "target_alignment"])
    if hasattr(wandb, "define_metric"):
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    ensemble_extractor, models = get_models(cfg)
    ensemble_loss = get_ensemble_loss(cfg, models)

    transform_fn = build_image_transform(cfg)
    source_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)
    source_loader = torch.utils.data.DataLoader(
        source_data, batch_size=cfg.data.batch_size, shuffle=False
    )
    target_loader = torch.utils.data.DataLoader(
        target_data, batch_size=cfg.data.batch_size, shuffle=False
    )

    source_crop = build_source_crop(cfg)
    target_crop = build_target_crop(cfg)
    gsdm_generator = GSDMGenerator(
        device=cfg.model.device,
        box_threshold=cfg.gsdm.box_threshold,
        text_threshold=cfg.gsdm.text_threshold,
        mask_fusion=cfg.gsdm.mask_fusion,
    )
    config_hash = hash_training_config(cfg)

    total = max(1, cfg.data.num_samples // cfg.data.batch_size)
    for img_index, ((image_org, _, path_org), (image_tgt, _, _)) in enumerate(
        zip(source_loader, target_loader)
    ):
        if cfg.data.batch_size * (img_index + 1) > cfg.data.num_samples:
            break
        print(f"\nProcessing M-Attack image {img_index + 1}/{total}")
        image_org = image_org.to(cfg.model.device)
        image_tgt = image_tgt.to(cfg.model.device)
        adv_image = target_attack(
            cfg,
            ensemble_extractor,
            ensemble_loss,
            source_crop,
            target_crop,
            img_index,
            image_org,
            image_tgt,
            gsdm_generator,
        )
        save_outputs(cfg, config_hash, adv_image, path_org)

    if hasattr(wandb, "finish"):
        wandb.finish()


if __name__ == "__main__":
    main()

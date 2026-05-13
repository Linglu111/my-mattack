import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from typing import List, Dict, Optional
from torch import nn
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from config_schema import MainConfig

from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
)
from surrogates.gsdm_generator import GSDMGenerator

from utils import hash_training_config, setup_wandb, ensure_dir

# 模型名称 → 特征提取器类的映射表
# 用于根据配置文件中的字符串名称动态实例化对应的CLIP模型
# - L336: CLIP ViT-L/14@336px (大模型，高分辨率)
# - B16:  CLIP ViT-B/16 (小模型，速度快)
# - B32:  CLIP ViT-B/32 (小模型，更粗粒度)
# - Laion: CLIP在LAION数据集上微调的版本
BACKBONE_MAP: Dict[str, type] = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}


def get_models(cfg: MainConfig):
    """根据配置初始化并加载代理模型（CLIP特征提取器）

    Args:
        cfg: 配置对象，包含 model.backbone（模型名称列表）、
             model.ensemble（是否使用集成）、model.device（设备）等

    Returns:
        tuple: (ensemble_extractor, models)
            - ensemble_extractor: 集成特征提取器（EnsembleFeatureExtractor）或单个模型
            - models: 所有模型实例的列表，用于后续创建集成损失函数

    Raises:
        ValueError: 当 ensemble=False 但指定了多个backbone时抛出异常

    逻辑说明：
        1. 校验配置一致性：非集成模式只能指定一个模型
        2. 遍历backbone列表，通过BACKBONE_MAP找到对应类并实例化
        3. 每个模型设为eval模式、移至指定设备、冻结参数（requires_grad=False）
        4. 根据ensemble标志决定返回集成包装器还是单个模型
    """
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
            )
        model_class = BACKBONE_MAP[backbone_name]
        # 实例化模型 → eval模式(关闭Dropout等) → 移至GPU/CPU → 冻结参数
        model = model_class().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)

    if cfg.model.ensemble:
        # 集成模式：用EnsembleFeatureExtractor包装所有模型
        # 前向传播时会依次调用每个模型，返回字典形式的特征
        ensemble_extractor = EnsembleFeatureExtractor(models)
    else:
        # 单模型模式：直接使用第一个模型
        ensemble_extractor = models[0]

    return ensemble_extractor, models


def get_ensemble_loss(cfg: MainConfig, models: List[nn.Module]):
    # 将多个模型传入EnsembleFeatureLoss，用于计算对抗图像特征与目标特征的余弦相似度损失
    ensemble_loss = EnsembleFeatureLoss(models)
    return ensemble_loss


def set_environment(seed=2023):
    """设置全局随机种子，确保实验可复现

    Args:
        seed (int): 随机种子值，默认2023

    固定以下随机数生成器：
        - Python内置random模块
        - PYTHONHASHSEED环境变量（影响字符串hash随机化）
        - NumPy随机数生成器
        - PyTorch CPU随机数生成器
        - PyTorch CUDA随机数生成器
        - cuDNN设为确定性模式（可能降低性能但保证复现）
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(pic):
    """将PIL.Image对象转换为PyTorch张量

    Args:
        pic (PIL.Image): PIL图像对象，支持多种模式(RGB, L, I, I;16, F等)

    Returns:
        torch.Tensor: 形状为 [C, H, W] 的张量，dtype与默认dtype一致

    注意：
        这是手动实现的转换，不使用transforms.ToTensor()。
        原因：需要支持更多PIL图像模式，且保持原始数值范围（不归一化到[0,1]）
    """
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """自定义数据集类，继承自ImageFolder

    与父类的区别：__getitem__额外返回图像的文件路径（第3个元素）
    这样在保存对抗样本时可以知道原始文件名和目录结构
    """

    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    set_environment()

    # Initialize wandb using shared utility
    setup_wandb(cfg, tags=["image_generation"])
    # Define metrics relationship for logging multiple images
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    # ---- 步骤1: 初始化代理模型 ----
    ensemble_extractor, models = get_models(cfg)
    ensemble_loss = get_ensemble_loss(cfg, models)

    # ---- 步骤2: 图像预处理 ----
    transform_fn = transforms.Compose(
        [
            transforms.Resize(
                cfg.model.input_res,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(cfg.model.input_res),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )

    # ---- 步骤3: 加载数据集 ----
    clean_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)

    data_loader_imagenet = torch.utils.data.DataLoader(
        clean_data, batch_size=cfg.data.batch_size, shuffle=False
    )
    data_loader_target = torch.utils.data.DataLoader(
        target_data, batch_size=cfg.data.batch_size, shuffle=False
    )

    print("Using source crop:", cfg.model.use_source_crop)
    print("Using target crop:", cfg.model.use_target_crop)

    # ---- 步骤4: 配置Local Matching裁剪策略 ----
    source_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_source_crop
        else torch.nn.Identity()
    )
    target_crop = (
        transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
        if cfg.model.use_target_crop
        else torch.nn.Identity()
    )

    # ---- 步骤5: 主循环 — 逐对处理图像 ----
    
    # DCA攻击时，统一初始化GSDM生成器（只初始化一次，避免重复加载模型）
    gsdm_generator = None
    if cfg.attack == "dca":
        gsdm_cfg = cfg.dca.gsdm if hasattr(cfg.dca, 'gsdm') else cfg.dca
        gsdm_generator = GSDMGenerator(
            device=cfg.model.device,
            box_threshold=getattr(gsdm_cfg, 'box_threshold', 0.25),
            text_threshold=getattr(gsdm_cfg, 'text_threshold', 0.20),
            mask_fusion=getattr(gsdm_cfg, 'mask_fusion', 'union'),
        )
    
    # 统一参数：所有攻击函数接受相同参数集，path_org和gsdm_generator在不需要时忽略
    attack_fn = {
        "fgsm": fgsm_attack,
        "mifgsm": mifgsm_attack,
        "pgd": pgd_attack,
        "dca": dca_attack,
    }[cfg.attack]
    
    config_hash = hash_training_config(cfg)
    
    for i, ((image_org, _, path_org), (image_tgt, _, path_tgt)) in enumerate(
        zip(data_loader_imagenet, data_loader_target)
    ):
        if cfg.data.batch_size * (i + 1) > cfg.data.num_samples:
            break

        print(f"\nProcessing image {i+1}/{cfg.data.num_samples//cfg.data.batch_size}")

        attack_imgpair(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_crop=source_crop,
            target_crop=target_crop,
            img_index=i,
            image_org=image_org,
            path_org=path_org,
            image_tgt=image_tgt,
            attack_fn=attack_fn,
            gsdm_generator=gsdm_generator,
            config_hash=config_hash,
        )

    wandb.finish()


def attack_imgpair(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    path_org: List[str],
    image_tgt: torch.Tensor,
    attack_fn,
    gsdm_generator=None,
    config_hash: str = "",
):
    image_org, image_tgt = image_org.to(cfg.model.device), image_tgt.to(cfg.model.device)

    adv_image, masks = attack_fn(
        cfg=cfg,
        ensemble_extractor=ensemble_extractor,
        ensemble_loss=ensemble_loss,
        source_crop=source_crop,
        target_crop=target_crop,
        img_index=img_index,
        image_org=image_org,
        image_tgt=image_tgt,
        gsdm_generator=gsdm_generator,
    )

    for path_idx in range(len(path_org)):
        folder = os.path.basename(os.path.dirname(path_org[path_idx]))
        name = os.path.basename(path_org[path_idx])
        folder_to_save = os.path.join(cfg.data.output, "img", config_hash, folder)
        ensure_dir(folder_to_save)

        ext = os.path.splitext(name)[1].lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
            save_name = os.path.splitext(name)[0] + ".png" if ext in [".jpg", ".jpeg"] else name
            torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, save_name))
        else:
            torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name))

        # 为每张图同步保存GSDM掩码可视化
        if masks and path_idx < len(masks):
            vis_dir = os.path.join(cfg.data.output, "visualization", config_hash, folder)
            ensure_dir(vis_dir)
            save_name_base = os.path.splitext(name)[0]
            save_gsdm_visualization(
                image_org[path_idx].detach(), masks[path_idx].detach(), adv_image[path_idx].detach(),
                os.path.join(vis_dir, f"{save_name_base}_gsdm.png")
            )


def log_metrics(pbar, metrics, img_index, epoch=None):
    """
    Log metrics to progress bar and wandb.

    Args:
        pbar: tqdm progress bar to update
        metrics: Dictionary of metrics to log
        img_index: Index of the image (for wandb logging)
        epoch: Optional epoch number for logging
    """
    # Format metrics for progress bar
    pbar_metrics = {
        k: f"{v:.5f}" if "sim" in k else f"{v:.3f}" for k, v in metrics.items()
    }
    pbar.set_postfix(pbar_metrics)

    # Prepare wandb metrics with image index
    wandb_metrics = {f"img{img_index}_{k}": v for k, v in metrics.items()}
    if epoch is not None:
        wandb_metrics["epoch"] = epoch

    # Log to wandb
    wandb.log(wandb_metrics)


def fgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
    gsdm_generator=None,
):
    """
    Perform FGSM attack on the image to generate adversarial examples.

    Args:
        cfg: Configuration parameters
        ensemble_extractor: Ensemble feature extractor model
        ensemble_loss: Ensemble loss function
        source_crop: Optional transform for cropping source images
        target_crop: Optional transform for cropping target images
        i: Index of the image (for logging)
        image_org: Original source image tensor
        image_tgt: Target image tensor to match features with

    Returns:
        torch.Tensor: Generated adversarial image
    """
    # 初始化扰动
    # 创建一个与image_org形状完全相同的全零张量，即需要优化的扰动
    delta = torch.zeros_like(image_org, requires_grad=True)

    # Progress bar for optimization
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            # target_crop(image_tgt)对目标图像进行随即裁剪
            # set_ground_truth提取目标图像的特征并保存，由于后续计算余弦相似度
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        # Forward pass
        adv_image = image_org + delta   # 构建当前轮次扰动
        adv_features = ensemble_extractor(adv_image)    # 提取对抗图像的特征

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)    # 计算对抗图像特征与目标图像特征的余弦相似度
        metrics["global_similarity"] = global_sim.item()

        if cfg.model.use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)  #对对抗图像进行随即裁剪
            local_features = ensemble_extractor(local_cropped)  # 提取局部特征
            local_sim = ensemble_loss(local_features)   # 计算局部相似度
            loss = local_sim
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = global_sim

        # Log current metrics
        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]  #计算梯度

        # Update delta using FGSM
        # clamp(delta, [-ε, ε]) 投影到约束空间
        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    # Create final adversarial image
    adv_image = image_org + delta   
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)    #归一化

    # Log final perturbation metrics
    final_metrics = {
        "max_delta": torch.max(torch.abs(delta)).item(),
        "mean_delta": torch.mean(torch.abs(delta)).item(),
    }
    log_metrics(pbar, final_metrics, img_index)

    return adv_image, []


def mifgsm_attack(
    cfg, ensemble_extractor, ensemble_loss,
    source_crop, target_crop, img_index,
    image_org, image_tgt, gsdm_generator=None,
):
    raise NotImplementedError("MI-FGSM attack not yet implemented. Use 'dca' for GSDM-based attack.")


def pgd_attack(
    cfg, ensemble_extractor, ensemble_loss,
    source_crop, target_crop, img_index,
    image_org, image_tgt, gsdm_generator=None,
):
    raise NotImplementedError("PGD attack not yet implemented. Use 'dca' for GSDM-based attack.")



def dca_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
    gsdm_generator=None,
):
    if gsdm_generator is None:
        raise ValueError("DCA attack requires GSDMGenerator")

    geo_label = getattr(cfg.dca, 'geo_label', None)

    batch_size = image_org.size(0)
    masks = []
    for b in range(batch_size):
        result = gsdm_generator.generate_mask(image_org[b], geo_label, return_visualization=True)
        if isinstance(result, tuple):
            mask, vis_data = result
            if vis_data.get("fallback"):
                print(f"  [GSDM] Image {b}: no geo-features detected → uniform mask (standard attack)")
        else:
            mask = result
        masks.append(mask)
    M = torch.stack(masks, dim=0).to(cfg.model.device)

    delta = torch.zeros_like(image_org, requires_grad=True)

    lpips_model = None
    if hasattr(cfg, 'dca') and cfg.dca.use_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net='alex').to(cfg.model.device)
            lpips_model.eval()
        except ImportError:
            print("Warning: lpips not installed, skipping perceptual loss")

    pbar = tqdm(range(cfg.optim.steps), desc=f"DCA Attack progress")

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
        loss = -global_sim  # 无向攻击

        if lpips_model is not None:
            adv_normalized = torch.clamp(adv_image / 255.0, 0.0, 1.0)
            org_normalized = torch.clamp(image_org / 255.0, 0.0, 1.0)
            loss_lpips = lpips_model(adv_normalized, org_normalized).mean()
            loss = loss + cfg.dca.lpips_weight * loss_lpips
            metrics["lpips_loss"] = loss_lpips.item()

        if cfg.model.use_source_crop:
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = loss - local_sim
            metrics["local_similarity"] = local_sim.item()

        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]
        M_expanded = M.unsqueeze(1).expand_as(delta)

        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad) * M_expanded,
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    final_metrics = {
        "max_delta": torch.max(torch.abs(delta)).item(),
        "mean_delta": torch.mean(torch.abs(delta)).item(),
    }
    log_metrics(pbar, final_metrics, img_index)

    return adv_image, masks


def save_gsdm_visualization(image_org, mask, adv_image, save_path):
    """保存GSDM掩码可视化结果"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image_org.cpu().permute(1, 2, 0).numpy() / 255.0)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(mask.cpu().numpy(), cmap='hot')
    axes[1].set_title("GSDM Mask (Geo-Saliency)")
    axes[1].axis('off')
    
    axes[2].imshow(adv_image.cpu().permute(1, 2, 0).numpy())
    axes[2].set_title("Adversarial Image")
    axes[2].axis('off')
    
    perturbation = (adv_image - image_org / 255.0).cpu().permute(1, 2, 0).numpy()
    perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
    axes[3].imshow(perturbation)
    axes[3].set_title("Perturbation (scaled)")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()

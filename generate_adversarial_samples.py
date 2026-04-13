import os
import json
import hashlib
import random
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
from PIL import Image
import hydra
from omegaconf import DictConfig
import os
from config_schema import MainConfig
from functools import partial
from typing import List, Dict, Optional
from torch import nn
from pytorch_lightning import seed_everything
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
)

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
    # source_crop: 对对抗图像进行随机裁剪
    # target_crop: 对目标图像进行随机裁剪
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
            img_index=i,
            image_org=image_org,
            path_org=path_org,
            image_tgt=image_tgt,
            target_crop=target_crop,
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
):
    image_org, image_tgt = image_org.to(cfg.model.device), image_tgt.to(
        cfg.model.device
    )
    attack_type = cfg.attack
    attack_fn = {
        "fgsm": fgsm_attack,
        "mifgsm": mifgsm_attack,
        "pgd": pgd_attack,
    }[attack_type]
    adv_image = attack_fn(
        cfg=cfg,
        ensemble_extractor=ensemble_extractor,
        ensemble_loss=ensemble_loss,
        source_crop=source_crop,
        target_crop=target_crop,
        img_index=img_index,
        image_org=image_org,
        image_tgt=image_tgt,
    )

    # Get config hash for output directory
    config_hash = hash_training_config(cfg)

    # Save images
    for path_idx in range(len(path_org)):
        # folder, name = (
        #     path_org[path_idx].split("/")[-2],
        #     path_org[path_idx].split("/")[-1],
        # )
        # 使用os.path模块来正确处理路径，而不是简单的字符串分割
        folder = os.path.basename(os.path.dirname(path_org[path_idx]))
        name = os.path.basename(path_org[path_idx])
        
        # Use config hash in output path
        folder_to_save = os.path.join(cfg.data.output, "img", config_hash, folder)
        ensure_dir(folder_to_save)

        # Get file extension and make it lowercase for case-insensitive check
        ext = os.path.splitext(name)[1].lower()
        
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
            # For JPEG files, convert to PNG
            if ext in [".jpg", ".jpeg"]:
                save_name = os.path.splitext(name)[0] + ".png"
            else:
                save_name = name
            
            torchvision.utils.save_image(
                adv_image[path_idx], os.path.join(folder_to_save, save_name)
            )
        else:
            # Save with original extension if not recognized
            torchvision.utils.save_image(
                adv_image[path_idx], os.path.join(folder_to_save, name)
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

    return adv_image


def mifgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    """
    Perform MI-FGSM attack on the image to generate adversarial examples.

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
    # # 初始化扰动
    # 创建一个与image_org形状完全相同的全零张量，即需要优化的扰动
    delta = torch.zeros_like(image_org, requires_grad=True) 
    momentum = torch.zeros_like(image_org, requires_grad=False) #动量张量

    # Progress bar for optimization
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        # Forward pass
        adv_image = image_org + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)    #计算对抗图像与目标图像的全局余弦相似度
        metrics["global_similarity"] = global_sim.item()

        if cfg.model.use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)  #局部随即裁剪
            local_features = ensemble_extractor(local_cropped)  #提取局部特征
            local_sim = ensemble_loss(local_features)   #计算局部余弦相似度
            loss = local_sim
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = global_sim

        log_metrics(pbar, metrics, img_index, epoch)

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]  #计算梯度

        # MI-FGSM update
        momentum = momentum * 0.9 + grad    #使用动量更新
        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(momentum),
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

    return adv_image


def pgd_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[transforms.RandomResizedCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
):
    """
    Perform PGD attack on the image to generate adversarial examples.

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
    # Initialize perturbation and momentum
    delta = torch.zeros_like(image_org, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg.optim.alpha)   #Adam优化器

    # Progress bar for optimization
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        # Forward pass
        adv_image = image_org + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)
        metrics["global_similarity"] = global_sim.item()

        if cfg.model.use_source_crop:
            # If using source crop, calculate additional local similarity
            local_cropped = source_crop(adv_image)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = -local_sim # since we want to maximize the loss 最小化负相似度
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = -global_sim

        log_metrics(pbar, metrics, img_index, epoch)

        optimizer.zero_grad()
        loss.backward()

        # PGD update
        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    # Create final adversarial image
    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    # Log final perturbation metrics
    final_metrics = {
        "max_delta": torch.max(torch.abs(delta)).item(),
        "mean_delta": torch.mean(torch.abs(delta)).item(),
    }
    log_metrics(pbar, final_metrics, img_index)

    return adv_image


if __name__ == "__main__":
    main()

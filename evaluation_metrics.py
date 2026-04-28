import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import os


def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算峰值信噪比 (Peak Signal-to-Noise Ratio)
    
    Args:
        img1: 原始图像 [C, H, W] 或 [B, C, H, W]
        img2: 对抗图像 [C, H, W] 或 [B, C, H, W]
        max_val: 像素最大值
    
    Returns:
        float: PSNR值 (dB)
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def calculate_ssim(img1, img2, window_size=11, max_val=1.0):
    """
    计算结构相似性指数 (Structural Similarity Index)
    
    Args:
        img1: 原始图像 [C, H, W] 或 [B, C, H, W]
        img2: 对抗图像 [C, H, W] 或 [B, C, H, W]
        window_size: 高斯窗口大小
        max_val: 像素最大值
    
    Returns:
        float: SSIM值 [-1, 1]
    """
    # 简化的SSIM实现
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def calculate_mask_coverage(mask, threshold=0.5):
    """
    计算掩码覆盖率
    
    Args:
        mask: 掩码张量 [H, W]，值域 [0, 1]
        threshold: 二值化阈值
    
    Returns:
        float: 掩码覆盖比例
    """
    binary_mask = (mask > threshold).float()
    coverage = binary_mask.sum() / mask.numel()
    return coverage.item()


def calculate_perturbation_localization(perturbation, mask, threshold=0.5):
    """
    计算扰动局部化程度：掩码区域内的扰动能量占比
    
    Args:
        perturbation: 扰动张量 [C, H, W]
        mask: 掩码张量 [H, W]，值域 [0, 1]
        threshold: 二值化阈值
    
    Returns:
        dict: 包含局部化指标的字典
    """
    # 计算扰动幅度
    pert_magnitude = torch.abs(perturbation).mean(dim=0)  # [H, W]
    
    # 二值化掩码
    binary_mask = (mask > threshold).float()
    
    # 掩码区域内的扰动能量
    masked_pert = pert_magnitude * binary_mask
    energy_in_mask = masked_pert.sum()
    total_energy = pert_magnitude.sum()
    
    # 局部化指标
    localization_ratio = (energy_in_mask / (total_energy + 1e-8)).item()
    
    # 掩码内平均扰动 vs 掩码外平均扰动
    mask_area = binary_mask.sum()
    non_mask_area = (1 - binary_mask).sum()
    
    avg_pert_in_mask = (energy_in_mask / (mask_area + 1e-8)).item()
    avg_pert_out_mask = ((total_energy - energy_in_mask) / (non_mask_area + 1e-8)).item()
    
    return {
        "localization_ratio": localization_ratio,
        "avg_pert_in_mask": avg_pert_in_mask,
        "avg_pert_out_mask": avg_pert_out_mask,
        "mask_coverage": (mask_area / mask.numel()).item(),
    }


def calculate_background_psnr(image_org, adv_image, mask, threshold=0.5):
    """
    计算背景（非掩码区域）的PSNR
    
    Args:
        image_org: 原始图像 [C, H, W]
        adv_image: 对抗图像 [C, H, W]
        mask: 掩码张量 [H, W]
        threshold: 二值化阈值
    
    Returns:
        float: 背景PSNR
    """
    binary_mask = (mask <= threshold).float()  # 背景区域
    
    # 只计算背景区域的MSE
    diff = (image_org - adv_image) ** 2
    diff_masked = diff * binary_mask.unsqueeze(0)
    
    mse = diff_masked.sum() / (binary_mask.sum() * image_org.size(0) + 1e-8)
    
    if mse == 0:
        return float('inf')
    
    max_val = 1.0 if image_org.max() <= 1.0 else 255.0
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse)).item()


def visualize_attack_results(image_org, adv_image, mask, save_path=None):
    """
    可视化攻击结果
    
    Args:
        image_org: 原始图像 [C, H, W]
        adv_image: 对抗图像 [C, H, W]
        mask: 掩码张量 [H, W]
        save_path: 保存路径
    
    Returns:
        matplotlib.figure.Figure: 图像对象
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 转换为numpy并调整维度
    if torch.is_tensor(image_org):
        img_org = image_org.cpu().permute(1, 2, 0).numpy()
        img_adv = adv_image.cpu().permute(1, 2, 0).numpy()
        mask_np = mask.cpu().numpy()
    else:
        img_org = image_org
        img_adv = adv_image
        mask_np = mask
    
    # 归一化到[0, 1]
    if img_org.max() > 1.0:
        img_org = img_org / 255.0
    if img_adv.max() > 1.0:
        img_adv = img_adv / 255.0
    
    # 原图
    axes[0, 0].imshow(np.clip(img_org, 0, 1))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    # 对抗图像
    axes[0, 1].imshow(np.clip(img_adv, 0, 1))
    axes[0, 1].set_title("Adversarial Image")
    axes[0, 1].axis('off')
    
    # GGM掩码
    im = axes[0, 2].imshow(mask_np, cmap='hot')
    axes[0, 2].set_title("GGM Mask")
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # 扰动（放大）
    perturbation = np.abs(img_adv - img_org)
    pert_scaled = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)
    axes[1, 0].imshow(pert_scaled)
    axes[1, 0].set_title("Perturbation (scaled)")
    axes[1, 0].axis('off')
    
    # 掩码叠加
    overlay = img_org.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] + mask_np * 0.5, 0, 1)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Mask Overlay")
    axes[1, 1].axis('off')
    
    # 局部化指标文本
    if torch.is_tensor(mask) and torch.is_tensor(adv_image) and torch.is_tensor(image_org):
        pert = adv_image - image_org
        metrics = calculate_perturbation_localization(pert, mask)
        psnr = calculate_psnr(image_org, adv_image, max_val=1.0)
        bg_psnr = calculate_background_psnr(image_org, adv_image, mask)
        
        text = f"PSNR: {psnr:.2f} dB\n"
        text += f"Background PSNR: {bg_psnr:.2f} dB\n"
        text += f"Localization Ratio: {metrics['localization_ratio']:.3f}\n"
        text += f"Mask Coverage: {metrics['mask_coverage']:.3f}\n"
        text += f"Avg Pert in Mask: {metrics['avg_pert_in_mask']:.4f}\n"
        text += f"Avg Pert out Mask: {metrics['avg_pert_out_mask']:.4f}"
    else:
        text = "Metrics require torch tensors"
    
    axes[1, 2].text(0.1, 0.5, text, fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title("Metrics")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    return fig


def evaluate_attack_comprehensive(image_org, adv_image, mask=None):
    """
    综合评估攻击效果
    
    Args:
        image_org: 原始图像 [C, H, W]
        adv_image: 对抗图像 [C, H, W]
        mask: 可选的GGM掩码 [H, W]
    
    Returns:
        dict: 包含所有评估指标的字典
    """
    metrics = {}
    
    # 基本图像质量指标
    metrics['psnr'] = calculate_psnr(image_org, adv_image, max_val=1.0)
    metrics['ssim'] = calculate_ssim(image_org, adv_image, max_val=1.0)
    
    # 扰动统计
    perturbation = adv_image - image_org
    metrics['max_perturbation'] = torch.abs(perturbation).max().item()
    metrics['mean_perturbation'] = torch.abs(perturbation).mean().item()
    metrics['l2_norm'] = torch.norm(perturbation).item()
    metrics['linf_norm'] = torch.abs(perturbation).max().item()
    
    # 如果提供了掩码，计算局部化指标
    if mask is not None:
        loc_metrics = calculate_perturbation_localization(perturbation, mask)
        metrics.update(loc_metrics)
        metrics['background_psnr'] = calculate_background_psnr(image_org, adv_image, mask)
    
    return metrics


if __name__ == "__main__":
    # 测试评估指标
    print("Testing evaluation metrics...")
    
    # 创建测试数据
    img1 = torch.rand(3, 224, 224)
    img2 = torch.clamp(img1 + torch.randn(3, 224, 224) * 0.01, 0, 1)
    mask = torch.rand(224, 224)
    
    # 测试各项指标
    print(f"PSNR: {calculate_psnr(img1, img2):.2f} dB")
    print(f"SSIM: {calculate_ssim(img1, img2):.4f}")
    print(f"Mask Coverage: {calculate_mask_coverage(mask):.4f}")
    
    loc = calculate_perturbation_localization(img2 - img1, mask)
    print(f"Localization Ratio: {loc['localization_ratio']:.4f}")
    
    bg_psnr = calculate_background_psnr(img1, img2, mask)
    print(f"Background PSNR: {bg_psnr:.2f} dB")
    
    # 综合评估
    all_metrics = evaluate_attack_comprehensive(img1, img2, mask)
    print("\nAll metrics:")
    for k, v in all_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\nEvaluation metrics test completed!")

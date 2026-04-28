import torch
import torch.nn.functional as F
import clip
from PIL import Image
import numpy as np


class GGMGenerator:
    """
    Geo-Gradient Mask Generator (GGM)
    
    利用CLIP跨模态梯度自动生成地理决策关键性掩码。
    核心思想：让CLIP"自己告诉自己"哪里是地理证据。
    """

    def __init__(self, device="cuda:0", sigma=3.0):
        """
        Args:
            device: 计算设备
            sigma: 高斯平滑核的标准差
        """
        self.device = device
        self.sigma = sigma
        # 加载CLIP模型
        self.model, self.preprocess = clip.load("ViT-B/16", device=device)
        self.model.eval()
        # 冻结CLIP参数
        for param in self.model.parameters():
            param.requires_grad = False

    def generate_mask(self, image_tensor, geo_label, return_visualization=False):
        """
        生成地理决策梯度掩码

        Args:
            image_tensor: 输入图像张量 [B, C, H, W] 或 [C, H, W]，值域 [0, 255]
            geo_label: 地理标签字符串（如 "New York, USA"）
            return_visualization: 是否返回可视化用的热力图

        Returns:
            mask: 连续权重掩码 [H, W]，值域 [0, 1]
            heatmap: (可选) 可视化热力图 [H, W, 3]
        """
        # 处理输入维度
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # 确保需要梯度
        image_input = image_tensor.clone().detach().requires_grad_(True)
        
        # Step 1: 构造地理提示文本
        text_prompt = f"A photo taken in {geo_label}"
        text_tokens = clip.tokenize([text_prompt]).to(self.device)
        
        # Step 2: 计算跨模态相似度并反向传播
        with torch.enable_grad():
            # 图像编码
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 文本编码
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算余弦相似度
            similarity = (image_features @ text_features.T).squeeze()
            
            # 反向传播获取梯度
            similarity.backward()
            
            # 获取梯度并取绝对值
            grad = image_input.grad  # [B, C, H, W]
            G = torch.abs(grad).squeeze(0)  # [C, H, W]
        
        # Step 3: 生成连续空间掩码
        # 通道平均
        G_avg = G.mean(dim=0)  # [H, W]
        
        # 高斯平滑
        G_smooth = self._gaussian_smooth(G_avg)
        
        # Min-Max归一化
        M = (G_smooth - G_smooth.min()) / (G_smooth.max() - G_smooth.min() + 1e-8)
        
        if return_visualization:
            heatmap = self._generate_heatmap(M)
            return M, heatmap
        
        return M

    def _gaussian_smooth(self, tensor):
        """对张量进行高斯平滑"""
        # 添加batch和channel维度 [1, 1, H, W]
        x = tensor.unsqueeze(0).unsqueeze(0)
        
        # 计算高斯核大小 (6*sigma + 1 确保覆盖99%的能量)
        kernel_size = int(6 * self.sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 生成2D高斯核
        sigma_tensor = torch.tensor(self.sigma, device=self.device)
        kernel = self._gaussian_kernel_2d(kernel_size, sigma_tensor).to(self.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        
        # 应用高斯滤波
        smoothed = F.conv2d(x, kernel, padding=kernel_size//2)
        return smoothed.squeeze(0).squeeze(0)

    def _gaussian_kernel_2d(self, kernel_size, sigma):
        """生成2D高斯核"""
        # 创建坐标网格
        x = torch.arange(kernel_size, dtype=torch.float32, device=sigma.device)
        x = x - (kernel_size - 1) / 2
        
        # 计算高斯分布
        gauss = torch.exp(-x.pow(2) / (2 * sigma.pow(2)))
        gauss = gauss / gauss.sum()
        
        # 创建2D核
        kernel_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        return kernel_2d

    def _generate_heatmap(self, mask):
        """生成可视化热力图"""
        # 将掩码转换为RGB热力图
        mask_np = mask.cpu().numpy()
        
        # 使用jet colormap
        import matplotlib.cm as cm
        cmap = cm.get_cmap('jet')
        heatmap = cmap(mask_np)[:, :, :3]  # [H, W, 3]
        
        return torch.from_numpy(heatmap).float().to(self.device)

    def apply_mask_to_perturbation(self, perturbation, mask):
        """
        将掩码应用到扰动上
        
        Args:
            perturbation: 扰动张量 [B, C, H, W] 或 [C, H, W]
            mask: 掩码张量 [H, W]
        
        Returns:
            masked_perturbation: 掩码后的扰动
        """
        if perturbation.dim() == 3:
            perturbation = perturbation.unsqueeze(0)
        
        # 扩展掩码维度 [1, 1, H, W]
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)
        
        # 逐元素相乘
        masked_perturbation = perturbation * mask_expanded
        
        return masked_perturbation.squeeze(0) if perturbation.size(0) == 1 else masked_perturbation


def test_ggm():
    """测试GGM生成器"""
    import torchvision.transforms as transforms
    from PIL import Image
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ggm = GGMGenerator(device=device, sigma=3.0)
    
    # 加载测试图像
    img_path = "myresources\images\img2gps3k\im2gps3ktest\31700873_d7c4159106_22_25159586@N00.jpg"
    image = Image.open(img_path).convert("RGB")
    
    # 转换为张量 [C, H, W]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).to(device) * 255.0  # 转换到[0, 255]
    
    # 生成掩码
    mask, heatmap = ggm.generate_mask(
        image_tensor, 
        geo_label="New York, USA",
        return_visualization=True
    )
    
    print(f"Mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"Mask mean: {mask.mean():.4f}")
    
    # 保存可视化结果
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(image_tensor.cpu().permute(1, 2, 0).numpy() / 255.0)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 掩码
    axes[1].imshow(mask.cpu().numpy(), cmap='gray')
    axes[1].set_title("GGM Mask")
    axes[1].axis('off')
    
    # 热力图
    axes[2].imshow(heatmap.cpu().numpy())
    axes[2].set_title("Heatmap")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("ggm_test_result.png", dpi=150, bbox_inches='tight')
    print("Saved visualization to ggm_test_result.png")


if __name__ == "__main__":
    test_ggm()

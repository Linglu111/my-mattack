import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path


class GSDMGenerator:
    """
    Geo-Saliency Detection and Masking (GSDM) Generator

    基于 DINOv2 自监督特征 + SAM 细化，实现无需标签训练的显著性区域检测与像素级掩码生成。

    工作流程：
      1. DINOv2 提取 patch 级特征 F ∈ R^(N×D)
      2. 计算每个 patch 与全局平均特征的余弦距离 → 显著性热力图
      3. OTSU 自适应阈值 → 二值化掩码
      4. SAM 以显著性掩码为 prompt 细化至像素级精度

    典型用法：
        gsdm = GSDMGenerator(device="cuda:0")
        mask = gsdm.generate_mask(image_tensor)  # image_tensor: [C, H, W], [0, 255]
    """

    DINO_MODEL_MAP = {
        "dinov2_vits14": "facebook/dinov2-small",
        "dinov2_vitb14": "facebook/dinov2-base",
        "dinov2_vitl14": "facebook/dinov2-large",
        "dinov2_vitg14": "facebook/dinov2-giant",
    }

    def __init__(
        self,
        device="cuda:0",
        dinov2_model="dinov2_vits14",
        sam_checkpoint=None,
        use_sam_refine=True,
    ):
        """
        Args:
            device: 计算设备，如 "cuda:0" 或 "cpu"
            dinov2_model: DINOv2 模型版本，可选 dinov2_vits14 / vitb14 / vitl14 / vitg14
            sam_checkpoint: SAM 模型权重本地路径，None 则自动下载
            use_sam_refine: 是否启用 SAM 像素级细化
        """
        self.device = device
        self.dinov2_model_name = dinov2_model
        self.use_sam_refine = use_sam_refine

        self._load_dinov2()
        if use_sam_refine:
            self._load_sam(sam_checkpoint)

    def _load_dinov2(self):
        from transformers import AutoImageProcessor, AutoModel

        model_id = self.DINO_MODEL_MAP.get(
            self.dinov2_model_name, self.dinov2_model_name
        )
        self.dino_processor = AutoImageProcessor.from_pretrained(model_id)
        self.dino_model = AutoModel.from_pretrained(model_id).to(self.device)
        self.dino_model.eval()

        self.patch_size = getattr(self.dino_model.config, "patch_size", 14)
        self.num_registers = getattr(
            self.dino_model.config, "num_register_tokens", 0
        )

    def _load_sam(self, checkpoint_path=None):
        from segment_anything import sam_model_registry, SamPredictor

        if checkpoint_path is None:
            checkpoint_path = self._download_sam_checkpoint()

        model_type = "vit_h"
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam_model = self.sam_model.to(self.device)
        self.sam_model.eval()
        self.sam_predictor = SamPredictor(self.sam_model)

    def _download_sam_checkpoint(self):
        cache_dir = Path.home() / ".cache" / "gsdm_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = cache_dir / "sam_vit_h_4b8939.pth"

        if checkpoint_path.exists():
            return str(checkpoint_path)

        import urllib.request

        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        print(f"Downloading SAM checkpoint from {url} ...")
        print(f"Saving to {checkpoint_path}")

        urllib.request.urlretrieve(url, str(checkpoint_path))
        print("Download complete.")
        return str(checkpoint_path)

    def generate_mask(self, image_tensor, geo_label=None, return_visualization=False):
        """
        生成地理显著性掩码。

        Args:
            image_tensor: 输入图像张量，形状 [C, H, W] 或 [B, C, H, W]，值域 [0, 255]
            geo_label: 保留接口兼容性，当前不使用
            return_visualization: 是否额外返回可视化数据字典

        Returns:
            mask: 连续权重掩码，形状 [H, W]，值域 [0, 1]
            vis_data: (可选) 包含检测框、置信度、实例数等的字典
        """
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        original_size = (image_tensor.size(2), image_tensor.size(1))
        pil_image = self._tensor_to_pil(image_tensor)

        with torch.no_grad():
            saliency_map = self._extract_saliency(pil_image)

        binary_mask = self._otsu_threshold(saliency_map)

        if binary_mask.sum() < 50:
            mask = torch.ones(original_size[::-1], device=self.device)
            if return_visualization:
                return mask, {
                    "saliency_map": saliency_map,
                    "binary_mask": binary_mask,
                    "num_salient_pixels": int(binary_mask.sum()),
                    "fallback": True,
                }
            return mask

        if self.use_sam_refine:
            with torch.no_grad():
                mask = self._sam_refine(pil_image, binary_mask)
        else:
            mask = torch.from_numpy(binary_mask.astype(np.float32)).to(self.device)

        if mask.shape != (original_size[1], original_size[0]):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        mask = torch.clamp(mask, 0.0, 1.0)

        if return_visualization:
            vis_data = {
                "saliency_map": saliency_map,
                "binary_mask": binary_mask,
                "num_salient_pixels": int(binary_mask.sum()),
            }
            return mask, vis_data

        return mask

    def _extract_saliency(self, pil_image):
        """
        使用 DINOv2 提取 patch 级特征并计算显著性热力图。

        步骤：
          1. DINOv2 前向传播，获取 patch tokens
          2. 计算全局平均特征 f̄
          3. 每个 patch 的余弦距离 s_i = 1 - cos(f_i, f̄)
          4. 重排为空间热力图，双线性插值回原图分辨率

        Args:
            pil_image: PIL.Image，RGB 模式

        Returns:
            saliency_map: np.ndarray [H, W]，值域 [0, 1]，亮度越高越"与众不同"
        """
        inputs = self.dino_processor(images=pil_image, return_tensors="pt").to(
            self.device
        )
        outputs = self.dino_model(**inputs)

        last_hidden_state = outputs.last_hidden_state
        patch_features = last_hidden_state[:, 1 + self.num_registers :, :]

        batch_size, num_patches, hidden_dim = patch_features.shape

        _, _, img_h, img_w = inputs.pixel_values.shape
        num_patches_h = img_h // self.patch_size
        num_patches_w = img_w // self.patch_size

        assert num_patches_h * num_patches_w == num_patches, (
            f"Patch count mismatch: {num_patches_h}*{num_patches_w}={num_patches_h * num_patches_w} "
            f"!= {num_patches}"
        )

        global_mean = patch_features.mean(dim=1, keepdim=True)
        patch_features_norm = F.normalize(patch_features, p=2, dim=-1)
        global_mean_norm = F.normalize(global_mean, p=2, dim=-1)

        cos_sim = (patch_features_norm * global_mean_norm).sum(dim=-1)
        saliency = 1.0 - cos_sim

        saliency_2d = saliency.view(batch_size, num_patches_h, num_patches_w)
        saliency_2d = saliency_2d.unsqueeze(1)

        saliency_resized = F.interpolate(
            saliency_2d,
            size=(pil_image.height, pil_image.width),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        saliency_np = saliency_resized.cpu().numpy()
        saliency_min = saliency_np.min()
        saliency_max = saliency_np.max()
        if saliency_max > saliency_min:
            saliency_np = (saliency_np - saliency_min) / (saliency_max - saliency_min)
        else:
            saliency_np = np.zeros_like(saliency_np)

        return saliency_np

    def _otsu_threshold(self, saliency_map):
        """
        对显著性热力图应用 OTSU 自适应阈值，生成二值化掩码。

        Args:
            saliency_map: np.ndarray [H, W]，值域 [0, 1]

        Returns:
            binary_mask: np.ndarray [H, W]，bool 类型
        """
        import cv2

        saliency_uint8 = (saliency_map * 255).astype(np.uint8)
        otsu_thresh, binary_mask = cv2.threshold(
            saliency_uint8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary_mask.astype(bool)

    def _sam_refine(self, pil_image, binary_mask):
        """
        使用 SAM 将显著性二值掩码细化为像素级精准掩码。

        将 DINOv2 + OTSU 生成的二值掩码作为 SAM 的 mask_input prompt，
        让 SAM 在 mask 约束下进行像素级分割细化。

        Args:
            pil_image: PIL.Image，RGB 模式
            binary_mask: np.ndarray [H, W]，bool 类型

        Returns:
            refined_mask: torch.Tensor [H, W]，值域 [0, 1]
        """
        import cv2

        image_np = np.array(pil_image.convert("RGB"))
        self.sam_predictor.set_image(image_np)

        mask_float = binary_mask.astype(np.float32)

        mask_256 = cv2.resize(
            mask_float, (256, 256), interpolation=cv2.INTER_LINEAR
        )

        mask_logits = (torch.from_numpy(mask_256).float().to(self.device) - 0.5) * 20.0
        mask_input = mask_logits.unsqueeze(0).unsqueeze(0)

        masks, scores, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=None,
            mask_input=mask_input,
            multimask_output=False,
        )

        refined = masks.squeeze(0).squeeze(0).float()
        return refined

    def _tensor_to_pil(self, tensor):
        """
        将 PyTorch 图像张量转换为 PIL Image。

        Args:
            tensor: [C, H, W]，值域 [0, 255]

        Returns:
            PIL.Image，RGB 模式
        """
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = torch.clamp(tensor, 0.0, 1.0)
        tensor = tensor.cpu()
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)


def test_gsdm():
    """
    本地快速测试：加载一张图像，生成 GSDM 掩码并保存可视化结果。

    输出五张子图并排展示：
      1. 原始图像
      2. DINOv2 显著性热力图
      3. OTSU 二值化掩码
      4. SAM 细化掩码
      5. 细化掩码叠加在原图上的半透明效果
    """
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    gsdm = GSDMGenerator(device=device, dinov2_model="dinov2_vits14")

    img_path = "myresources/images/img2gps3k/im2gps3ktest"
    png_files = []
    for root, _, files in os.walk(img_path):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                png_files.append(os.path.join(root, f))
    if png_files:
        img_path = png_files[0]
    else:
        print(f"No test image found in {img_path}!")
        return

    print(f"Loading image: {img_path}")
    image = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(device) * 255.0

    mask, vis_data = gsdm.generate_mask(image_tensor, return_visualization=True)

    print(f"Mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"Mask mean: {mask.mean():.4f}")
    print(f"Salient pixels (OTSU): {vis_data.get('num_salient_pixels', 0)}")

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(vis_data["saliency_map"], cmap="hot")
    axes[1].set_title("DINOv2 Saliency Map")
    axes[1].axis("off")

    axes[2].imshow(vis_data["binary_mask"], cmap="gray")
    axes[2].set_title("OTSU Binary Mask")
    axes[2].axis("off")

    axes[3].imshow(mask.cpu().numpy(), cmap="hot")
    axes[3].set_title("SAM Refined Mask")
    axes[3].axis("off")

    axes[4].imshow(image)
    axes[4].imshow(mask.cpu().numpy(), cmap="hot", alpha=0.5)
    axes[4].set_title("Overlay")
    axes[4].axis("off")

    os.makedirs("LAT/gsdm", exist_ok=True)
    save_path = "LAT/gsdm/gsdm_test_result.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    test_gsdm()

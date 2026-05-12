import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path


GEO_DETECT_CLASSES = (
    "building. architecture. street sign. traffic sign. landmark. "
    "storefront. billboard. bridge. statue. tower. temple. church. "
    "mosque. monument. fountain. sculpture. clock tower. dome. minaret. "
    "pagoda. archway. gate. pillar. column. facade."
)


class GSDMGenerator:
    """
    Geo-Saliency Detection and Masking (GSDM) Generator

    基于GroundingDINO + SAM的二阶管道，实现像素级精准的地理显著性掩码生成。
    工作流程：
    1. 检测阶段：GroundingDINO检测图像中所有具有地理意义的视觉元素
    2. 细化阶段：SAM将检测框转化为像素级精准分割掩码
    3. 融合阶段：将所有实例掩码融合为单一连续权重掩码
    """

    def __init__(
        self,
        device="cuda:0",
        box_threshold=0.25,
        text_threshold=0.20,
        detect_classes=None,
        mask_fusion="union",
        sam_checkpoint=None,
    ):
        """
        Args:
            device: 计算设备
            box_threshold: GroundingDINO检测框置信度阈值
            text_threshold: GroundingDINO文本匹配阈值
            detect_classes: 检测目标类别文本（用句号分隔），默认使用GEO_DETECT_CLASSES
            mask_fusion: 掩码融合策略 ["union", "weighted", "max"]
            sam_checkpoint: SAM模型权重路径，默认自动下载
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.detect_classes = detect_classes or GEO_DETECT_CLASSES
        self.mask_fusion = mask_fusion

        self._load_grounding_dino()
        self._load_sam(sam_checkpoint)

    def _load_grounding_dino(self):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = "IDEA-Research/grounding-dino-base"
        self.gd_processor = AutoProcessor.from_pretrained(model_id)
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        self.gd_model.eval()

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
        生成地理显著性掩码

        Args:
            image_tensor: 输入图像张量 [C, H, W] 或 [B, C, H, W]，值域 [0, 255]
            geo_label: 地理标签（可选，用于增强检测提示）
            return_visualization: 是否返回可视化数据

        Returns:
            mask: 连续权重掩码 [H, W]，值域 [0, 1]
            vis_data: (可选) 可视化数据字典
        """
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        original_size = (image_tensor.size(2), image_tensor.size(1))

        pil_image = self._tensor_to_pil(image_tensor)

        with torch.no_grad():
            boxes, confidences = self._detect_with_grounding_dino(pil_image)

        if len(boxes) == 0:
            mask = torch.zeros(original_size[::-1], device=self.device)
            if return_visualization:
                return mask, {"boxes": [], "mask": mask}
            return mask

        with torch.no_grad():
            instance_masks = self._segment_with_sam(pil_image, boxes, confidences)

        mask = self._fuse_masks(instance_masks, confidences)

        if mask.shape != (original_size[1], original_size[0]):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)

        if return_visualization:
            vis_data = {
                "boxes": boxes,
                "confidences": confidences,
                "num_instances": len(boxes),
                "mask": mask.clone(),
            }
            return mask, vis_data

        return mask

    def _detect_with_grounding_dino(self, pil_image):
        inputs = self.gd_processor(
            images=pil_image,
            text=self.detect_classes,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.gd_model(**inputs)
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )

        result = results[0]
        boxes = result["boxes"].cpu().numpy()
        confidences = result["scores"].cpu().numpy()

        return boxes, confidences

    def _segment_with_sam(self, pil_image, boxes, confidences):
        image_np = np.array(pil_image.convert("RGB"))
        self.sam_predictor.set_image(image_np)

        if len(boxes) == 0:
            return []

        input_boxes = torch.from_numpy(boxes).to(self.device)

        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            input_boxes, image_np.shape[:2]
        )

        masks, scores, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks = masks.squeeze(1)

        instance_masks = []
        for i in range(len(masks)):
            instance_masks.append(masks[i].float())

        return instance_masks

    def _fuse_masks(self, instance_masks, confidences):
        if len(instance_masks) == 0:
            return torch.zeros(
                self.sam_predictor.original_size, device=self.device
            )

        target_size = self.sam_predictor.original_size

        if self.mask_fusion == "union":
            fused = torch.zeros(target_size, device=self.device)
            for mask in instance_masks:
                resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                fused = torch.maximum(fused, resized)

        elif self.mask_fusion == "weighted":
            fused = torch.zeros(target_size, device=self.device)
            weights = torch.from_numpy(confidences).float().to(self.device)
            weights = weights / (weights.sum() + 1e-8)

            for mask, w in zip(instance_masks, weights):
                resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                fused = fused + w * resized

        elif self.mask_fusion == "max":
            stacked = []
            for mask in instance_masks:
                resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                stacked.append(resized)
            stacked = torch.stack(stacked, dim=0)
            fused = stacked.max(dim=0)[0]

        else:
            raise ValueError(f"Unknown mask_fusion strategy: {self.mask_fusion}")

        return torch.clamp(fused, 0.0, 1.0)

    def _tensor_to_pil(self, tensor):
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = torch.clamp(tensor, 0.0, 1.0)
        tensor = tensor.cpu()
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)


def test_gsdm():
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    gsdm = GSDMGenerator(device=device, box_threshold=0.25, text_threshold=0.20)

    img_path = "resources/images/bigscale/nips17/0.png"
    if not os.path.exists(img_path):
        img_path = "resources/images/bigscale/nips17"

        png_files = []
        for root, _, files in os.walk(img_path):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    png_files.append(os.path.join(root, f))
        if png_files:
            img_path = png_files[0]
        else:
            print("No test image found!")
            return

    print(f"Loading image: {img_path}")
    image = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(device) * 255.0

    mask, vis_data = gsdm.generate_mask(
        image_tensor, return_visualization=True
    )

    print(f"Mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"Mask mean: {mask.mean():.4f}")
    print(f"Number of detected instances: {vis_data['num_instances']}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask.cpu().numpy(), cmap="hot")
    axes[1].set_title("GSDM Mask (Geo-Saliency)")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(mask.cpu().numpy(), cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("gsdm_test_result.png", dpi=150, bbox_inches="tight")
    print("Saved visualization to gsdm_test_result.png")


if __name__ == "__main__":
    test_gsdm()

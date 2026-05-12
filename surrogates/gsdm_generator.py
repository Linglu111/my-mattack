import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path


# 开放式检测提示词定义
# GroundingDINO 是开放式词汇检测器，这里用自然语言描述所有与"地理"相关的
# 视觉元素类别。用句号分隔每个类别，模型会同时检测这些概念。
GEO_DETECT_CLASSES = (
    "building. architecture. street sign. traffic sign. landmark. "
    "storefront. billboard. bridge. statue. tower. temple. church. "
    "mosque. monument. fountain. sculpture. clock tower. dome. minaret. "
    "pagoda. archway. gate. pillar. column. facade. "
    "tree. forest. mountain. hill. river. lake. ocean. coastline. "
    "road. highway. intersection. crosswalk. traffic light. sidewalk. "
    "vehicle. car. bus. bicycle. boat. train. airplane. "
    "sign. banner. flag. mural. graffiti. poster. display. "
    "person. crowd. market stall. umbrella. bench. fence. wall."
)


class GSDMGenerator:
    """
    Geo-Saliency Detection and Masking (GSDM) Generator

    基于 GroundingDINO + SAM 的二阶管道，实现像素级精准的地理显著性掩码生成。
    工作流程：
      1. 检测阶段：GroundingDINO 以开放式词汇检测图像中所有地理相关视觉元素
      2. 细化阶段：SAM 将检测框转化为像素级精准分割掩码
      3. 融合阶段：将所有实例掩码融合为单一连续权重掩码

    典型用法：
        gsdm = GSDMGenerator(device="cuda:0")
        mask = gsdm.generate_mask(image_tensor)  # image_tensor: [C, H, W], [0, 255]
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
            device: 计算设备，如 "cuda:0" 或 "cpu"
            box_threshold: GroundingDINO 检测框置信度阈值，低于此值的框会被过滤
            text_threshold: GroundingDINO 文本-视觉对齐阈值，控制类别匹配严格度
            detect_classes: 检测目标类别文本（用句号分隔），默认使用 GEO_DETECT_CLASSES
            mask_fusion: 多实例掩码融合策略，可选 ["union", "weighted", "max"]
            sam_checkpoint: SAM 模型权重本地路径，None 则自动下载到 ~/.cache/gsdm_models
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.detect_classes = detect_classes or GEO_DETECT_CLASSES
        self.mask_fusion = mask_fusion

        # 加载两个子模型
        self._load_grounding_dino()
        self._load_sam(sam_checkpoint)


    # 模型加载
    def _load_grounding_dino(self):
        """加载 GroundingDINO 开放式词汇检测模型到指定设备。"""
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        model_id = "IDEA-Research/grounding-dino-base"
        # processor 负责图像预处理和后处理（包括文本编码）
        self.gd_processor = AutoProcessor.from_pretrained(model_id)
        # 主模型：输入图像+文本，输出检测框与类别对齐分数
        self.gd_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(self.device)
        self.gd_model.eval()

    def _load_sam(self, checkpoint_path=None):
        """加载 SAM (Segment Anything Model) 分割模型。"""
        from segment_anything import sam_model_registry, SamPredictor

        if checkpoint_path is None:
            checkpoint_path = self._download_sam_checkpoint()

        model_type = "vit_h"  # ViT-Huge，最高质量，约 2.4GB
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam_model = self.sam_model.to(self.device)
        self.sam_model.eval()
        # SamPredictor 提供高层 API：set_image() + predict()/predict_torch()
        self.sam_predictor = SamPredictor(self.sam_model)

    def _download_sam_checkpoint(self):
        """自动下载 SAM ViT-H 权重到本地缓存目录（如不存在）。"""
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

    # 主接口：掩码生成
    def generate_mask(self, image_tensor, geo_label=None, return_visualization=False):
        """
        生成地理显著性掩码。

        Args:
            image_tensor: 输入图像张量，形状 [C, H, W] 或 [B, C, H, W]，值域 [0, 255]
            geo_label: 地理标签文本（可选），可覆盖默认检测类别以增强提示
            return_visualization: 是否额外返回可视化数据字典

        Returns:
            mask: 连续权重掩码，形状 [H, W]，值域 [0, 1]
            vis_data: (可选) 包含检测框、置信度、实例数等的字典
        """
        # 去除 batch 维度，确保输入为单张图像 [C, H, W]
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)

        # 记录原始空间尺寸 (W, H)，后续用于将掩码插值回原尺寸
        original_size = (image_tensor.size(2), image_tensor.size(1))

        # 将 PyTorch 张量转为 PIL Image，供 GroundingDINO 和 SAM 使用
        pil_image = self._tensor_to_pil(image_tensor)

        # Stage 1: GroundingDINO 开放式检测
        with torch.no_grad():
            boxes, confidences = self._detect_with_grounding_dino(pil_image)

        # 未检出地理目标时：回退为均匀掩码（等价于标准MI-FGSM全局攻击）
        if len(boxes) == 0:
            mask = torch.ones(original_size[::-1], device=self.device)
            if return_visualization:
                return mask, {"boxes": [], "confidences": [], "num_instances": 0, "mask": mask, "fallback": True}
            return mask

        # Stage 2: SAM 将检测框细化为像素级掩码
        with torch.no_grad():
            instance_masks = self._segment_with_sam(pil_image, boxes, confidences)

        # Stage 3: 多实例掩码融合为单一连续掩码
        mask = self._fuse_masks(instance_masks, confidences)

        # SAM 内部会将图像 resize 到 1024x1024 处理，因此输出掩码尺寸可能与原图不同
        # 这里用双线性插值将掩码还原到原始图像尺寸
        if mask.shape != (original_size[1], original_size[0]):
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),   # [1, 1, H, W]
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

    # Stage 1: 检测
    def _detect_with_grounding_dino(self, pil_image):
        """
        使用 GroundingDINO 对单张图像进行开放式词汇检测。

        Args:
            pil_image: PIL.Image，RGB 模式

        Returns:
            boxes: ndarray [N, 4]，每个检测框的 [x1, y1, x2, y2] 坐标
            confidences: ndarray [N]，每个框的置信度分数
        """
        # processor 将图像和文本编码为模型输入格式
        inputs = self.gd_processor(
            images=pil_image,
            text=self.detect_classes,
            return_tensors="pt",
        ).to(self.device)

        # 前向传播：模型输出原始 logits
        outputs = self.gd_model(**inputs)

        # 后处理：将 logits 解码为实际检测框、分数和对应文本短语
        # target_sizes 用于将归一化坐标映射回像素坐标
        results = self.gd_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )

        result = results[0]  # 单张图像，取第一个结果
        boxes = result["boxes"].cpu().numpy()
        confidences = result["scores"].cpu().numpy()

        return boxes, confidences

    # Stage 2: 分割
    def _segment_with_sam(self, pil_image, boxes, confidences):
        """
        使用 SAM 将 GroundingDINO 输出的检测框细化为像素级分割掩码。

        Args:
            pil_image: PIL.Image，RGB 模式
            boxes: ndarray [N, 4]，检测框坐标
            confidences: ndarray [N]，检测框置信度（当前未用于加权分割，仅保留接口）

        Returns:
            instance_masks: List[Tensor]，每个元素为 SAM 输出的二值掩码 [H, W]
        """
        # SAM 要求 numpy 数组输入，通道顺序 HWC
        image_np = np.array(pil_image.convert("RGB"))
        self.sam_predictor.set_image(image_np)

        if len(boxes) == 0:
            return []

        # 将检测框转为 torch 张量并移到 GPU
        input_boxes = torch.from_numpy(boxes).to(self.device)

        # SAM 内部坐标变换：将原始图像坐标映射到 1024x1024 的输入空间
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(
            input_boxes, image_np.shape[:2]
        )

        # predict_torch 支持批量框输入，multimask_output=False 只返回最佳掩码
        masks, scores, _ = self.sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        # masks 形状: [N, 1, H, W] -> 去掉单掩码维度 -> [N, H, W]
        masks = masks.squeeze(1)

        # 拆分为列表，便于后续逐实例处理
        instance_masks = []
        for i in range(len(masks)):
            instance_masks.append(masks[i].float())

        return instance_masks

    # Stage 3: 掩码融合
    def _fuse_masks(self, instance_masks, confidences):
        """
        将多个实例掩码融合为单一连续权重掩码。

        支持三种策略：
          - union:  逐像素取最大值（宽松并集，任何被检出的区域都保留）
          - weighted: 按检测置信度加权求和（置信度高的实例贡献更大）
          - max:    与 union 等价，但实现方式不同（先堆叠再全局 max）

        Args:
            instance_masks: List[Tensor]，每个形状 [H, W]，值域 [0, 1]
            confidences: ndarray [N]，每个实例的检测置信度

        Returns:
            fused: Tensor [H, W]，值域 [0, 1]
        """
        if len(instance_masks) == 0:
            return torch.zeros(
                self.sam_predictor.original_size, device=self.device
            )

        # SAM 内部处理尺寸（通常是 1024x1024 或原始图像尺寸）
        target_size = self.sam_predictor.original_size

        if self.mask_fusion == "union":
            # 宽松并集：只要任一实例覆盖该像素，掩码值就接近 1
            fused = torch.zeros(target_size, device=self.device)
            for mask in instance_masks:
                # 将每个实例掩码插值到统一尺寸
                resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                fused = torch.maximum(fused, resized)

        elif self.mask_fusion == "weighted":
            # 加权融合：高置信度实例拥有更大权重，适合强调显著区域
            fused = torch.zeros(target_size, device=self.device)
            weights = torch.from_numpy(confidences).float().to(self.device)
            weights = weights / (weights.sum() + 1e-8)  # 归一化到概率分布

            for mask, w in zip(instance_masks, weights):
                resized = F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
                fused = fused + w * resized

        elif self.mask_fusion == "max":
            # 先堆叠所有掩码为 [N, H, W]，再沿实例维度取最大值
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

    # 工具方法
    def _tensor_to_pil(self, tensor):
        """
        将 PyTorch 图像张量转换为 PIL Image。

        处理逻辑：
          - 若最大值 > 1.0，假设值域为 [0, 255]，先归一化到 [0, 1]
          - 截断到合法范围，转到 CPU
          - 通道顺序：CHW -> HWC
          - 转为 uint8 numpy 数组后构建 PIL Image

        Args:
            tensor: [C, H, W]

        Returns:
            PIL.Image，RGB 模式
        """
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        tensor = torch.clamp(tensor, 0.0, 1.0)
        tensor = tensor.cpu()
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)


# 本地测试入口
def test_gsdm():
    """
    本地快速测试：加载一张图像，生成 GSDM 掩码并保存可视化结果。

    输出三张子图并排展示：
      1. 原始图像
      2. GSDM 掩码热力图
      3. 掩码叠加在原图上的半透明效果
    """
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    gsdm = GSDMGenerator(device=device, box_threshold=0.25, text_threshold=0.20)

    # 自动查找测试图像
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

    # ToTensor 输出 [0, 1]，乘 255 转为 GSDM 期望的 [0, 255] 值域
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).to(device) * 255.0

    mask, vis_data = gsdm.generate_mask(
        image_tensor, return_visualization=True
    )

    print(f"Mask shape: {mask.shape}")
    print(f"Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
    print(f"Mask mean: {mask.mean():.4f}")
    print(f"Number of detected instances: {vis_data['num_instances']}")

    # 可视化
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

    os.makedirs("LAT/gsdm", exist_ok=True)
    save_path = "LAT/gsdm/gsdm_test_result.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    test_gsdm()

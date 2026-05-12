from dataclasses import dataclass, field
from typing import Optional
from hydra.core.config_store import ConfigStore



@dataclass
class WandbConfig:
    """Wandb-specific configuration"""

    entity: str = "???"  # fill your wandb entity
    project: str = "local_adversarial_attack"


@dataclass
class BlackboxConfig:
    """Configuration for blackbox model evaluation"""

    model_name: str = "gpt4v"  # Can be gpt4v, claude, gemini, gpt_score
    batch_size: int = 1
    timeout: int = 30


@dataclass
class DataConfig:
    """Data loading configuration"""

    batch_size: int = 1
    num_samples: int = 100
    cle_data_path: str = "resources/images/bigscale"
    tgt_data_path: str = "resources/images/target_images"
    output: str = "./img_output"
    adv_img_dir: Optional[str] = None  # 直接指定对抗图像目录路径
    # # 自定义数据集
    # cle_data_path: str = "myresources/images/clean"
    # tgt_data_path: str = "myresources/images/target_images"


@dataclass
class OptimConfig:
    """Optimization parameters"""

    alpha: float = 1.0
    epsilon: int = 8
    steps: int = 300


@dataclass
class ModelConfig:
    """Model-specific parameters"""

    input_res: int = 336
    use_source_crop: bool = True
    use_target_crop: bool = True
    crop_scale: tuple = (0.5, 0.9)
    ensemble: bool = True
    device: str = "cuda:0"  # Can be "cpu", "cuda:0", "cuda:1", etc.
    backbone: list = field(default_factory=lambda: [
        "L336",
        "B16",
        "B32",
        "Laion",
    ])  # List of models to use: L336, B16, B32, Laion


@dataclass
class GSDMConfig:
    """GSDM (Geo-Saliency Detection and Masking) parameters"""

    box_threshold: float = 0.25  # GroundingDINO检测框置信度阈值
    text_threshold: float = 0.20  # GroundingDINO文本匹配阈值
    detect_classes: str = (
        "building. architecture. street sign. traffic sign. landmark. "
        "storefront. billboard. bridge. statue. tower. temple. church. "
        "mosque. monument. fountain. sculpture. clock tower. dome. minaret. "
        "pagoda. archway. gate. pillar. column. facade."
    )
    mask_fusion: str = "union"  # 掩码融合策略 [union, weighted, max]


@dataclass
class DCAConfig:
    """DCA (Decision-aware Cross-modal Attention Masking) parameters"""

    use_gsdm: bool = True  # 是否使用GSDM掩码（始终为True）
    gsdm: GSDMConfig = field(default_factory=GSDMConfig)
    use_lpips: bool = False  # 是否使用LPIPS感知损失
    lpips_weight: float = 0.2  # LPIPS损失权重
    geo_label: Optional[str] = None  # 地理标签（可选）


@dataclass
class MainConfig:
    """Main configuration combining all sub-configs"""

    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    blackbox: BlackboxConfig = field(default_factory=BlackboxConfig)
    dca: DCAConfig = field(default_factory=DCAConfig)
    attack: str = "fgsm"  # can be [fgsm, mifgsm, pgd, dca]


# register config for different setting
@dataclass
class Ensemble3ModelsConfig(MainConfig):
    """Configuration for ensemble_3models.py"""

    data: DataConfig = field(default_factory=lambda: DataConfig(batch_size=1))
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        use_source_crop=True, use_target_crop=True, backbone=["B16", "B32", "Laion"]
    ))


# Register configs with Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=MainConfig)
cs.store(name="ensemble_3models", node=Ensemble3ModelsConfig)

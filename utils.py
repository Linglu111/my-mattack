"""Shared utilities for adversarial attack and text generation models."""

import os
import json
import yaml
import hashlib
import base64
import random
from typing import Dict, List

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from torch import nn
import wandb

from config_schema import MainConfig


def load_api_keys() -> Dict[str, str]:
    """Load API keys from the api_keys file.
    
    Returns:
        Dict[str, str]: Dictionary containing API keys for different models
        
    Raises:
        FileNotFoundError: If no api_keys file is found
    """
    for ext in ['yaml', 'yml', 'json']:
        file_path = f'api_keys.{ext}'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                if ext in ['yaml', 'yml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
    
    raise FileNotFoundError(
        "API keys file not found. Please create api_keys.yaml, api_keys.yml, or api_keys.json "
        "in the root directory with your API keys."
    )


def get_api_key(model_name: str) -> str:
    """Get API key for specified model.
    
    Args:
        model_name: Name of the model to get API key for
        
    Returns:
        str: API key for the specified model
        
    Raises:
        KeyError: If API key for model is not found
    """
    api_keys = load_api_keys()
    if model_name not in api_keys:
        raise KeyError(
            f"API key for {model_name} not found in api_keys file. "
            f"Available models: {list(api_keys.keys())}"
        )
    return api_keys[model_name]


def set_environment(seed=2023):
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(pic):
    """Convert a PIL image to a tensor while preserving the original value range."""
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """ImageFolder variant that also returns the source file path."""

    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


def build_image_transform(cfg: MainConfig):
    return transforms.Compose(
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


def build_source_crop(cfg: MainConfig):
    if cfg.model.use_source_crop:
        return transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
    return torch.nn.Identity()


def build_target_crop(cfg: MainConfig):
    if cfg.model.use_target_crop:
        return transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale)
    return torch.nn.Identity()


def get_models(cfg: MainConfig):
    """Instantiate the configured CLIP surrogate models."""
    from surrogates import (
        ClipB16FeatureExtractor,
        ClipB32FeatureExtractor,
        ClipL336FeatureExtractor,
        ClipLaionFeatureExtractor,
        EnsembleFeatureExtractor,
    )

    backbone_map = {
        "L336": ClipL336FeatureExtractor,
        "B16": ClipB16FeatureExtractor,
        "B32": ClipB32FeatureExtractor,
        "Laion": ClipLaionFeatureExtractor,
    }

    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in backbone_map:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. Valid options are: {list(backbone_map.keys())}"
            )
        model = backbone_map[backbone_name]().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)

    if cfg.model.ensemble:
        return EnsembleFeatureExtractor(models), models
    return models[0], models


def get_ensemble_loss(cfg: MainConfig, models: List[nn.Module]):
    from surrogates import EnsembleFeatureLoss

    return EnsembleFeatureLoss(models)


def hash_training_config(cfg: MainConfig) -> str:
    """Create a deterministic hash of training-relevant config parameters.
    
    Args:
        cfg: Configuration object containing model settings
        
    Returns:
        str: MD5 hash of the config parameters
    """
    # Convert backbone list to plain Python list
    if isinstance(cfg.model.backbone, (list, tuple)):
        backbone = list(cfg.model.backbone)
    else:
        backbone = OmegaConf.to_container(cfg.model.backbone)
        
    # Create config dict with converted values
    attack_name = str(cfg.attack).lower()
    train_config = {
        "attack": attack_name,
        "data": {
            "batch_size": int(cfg.data.batch_size),
            "num_samples": int(cfg.data.num_samples),
            "cle_data_path": str(cfg.data.cle_data_path),
        },
        "optim": {
            "alpha": float(cfg.optim.alpha),
            "epsilon": int(cfg.optim.epsilon),
            "steps": int(cfg.optim.steps),
        },
        "model": {
            "input_res": int(cfg.model.input_res),
            "use_source_crop": bool(cfg.model.use_source_crop),
            "crop_scale": tuple(float(x) for x in cfg.model.crop_scale),
            "ensemble": bool(cfg.model.ensemble),
            "backbone": backbone,
        },
        "gsdm": {
            "box_threshold": float(cfg.gsdm.box_threshold),
            "text_threshold": float(cfg.gsdm.text_threshold),
            "mask_fusion": str(cfg.gsdm.mask_fusion),
            "geo_label": cfg.gsdm.geo_label,
        },
    }

    if attack_name == "gsdm":
        train_config["data"]["tgt_data_path"] = str(cfg.data.tgt_data_path)
        train_config["model"]["use_target_crop"] = bool(cfg.model.use_target_crop)
        train_config["gsdm"]["use_lpips"] = bool(cfg.gsdm.use_lpips)
        train_config["gsdm"]["lpips_weight"] = float(cfg.gsdm.lpips_weight)

    if attack_name == "hge":
        train_config["hge"] = {
            "enabled": bool(cfg.hge.enabled),
            "k_country": int(cfg.hge.k_country),
            "k_city": int(cfg.hge.k_city),
            "temperature": float(cfg.hge.temperature),
            "lambda_country": float(cfg.hge.lambda_country),
            "lambda_city": float(cfg.hge.lambda_city),
            "lambda_city_global_suppress": float(cfg.hge.lambda_city_global_suppress),
            "lambda_hge": float(cfg.hge.lambda_hge),
            "topk_suppress_weight": float(cfg.hge.topk_suppress_weight),
            "evidence_suppress_weight": float(cfg.hge.evidence_suppress_weight),
            "tv_weight": float(cfg.hge.tv_weight),
            "mask_epsilon": float(cfg.hge.mask_epsilon),
            "bg_epsilon": float(cfg.hge.bg_epsilon),
            "bg_lowfreq_enabled": bool(cfg.hge.bg_lowfreq_enabled),
            "bg_lowfreq_ratio": float(cfg.hge.bg_lowfreq_ratio),
            "fallback_mask_epsilon": float(cfg.hge.fallback_mask_epsilon),
            "high_epsilon_max_mask_area": float(cfg.hge.high_epsilon_max_mask_area),
            "context_dilation_kernel": int(cfg.hge.context_dilation_kernel),
            "context_padding_ratio": float(cfg.hge.context_padding_ratio),
            "vocab_path": cfg.hge.vocab_path,
        }
    
    # Convert to JSON string with sorted keys
    json_str = json.dumps(train_config, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


def setup_wandb(cfg: MainConfig, tags=None) -> None:
    """Initialize Weights & Biases logging.
    
    Args:
        cfg: Configuration object containing wandb settings
    """
    if not hasattr(wandb, "init"):
        print("Warning: wandb.init is unavailable; skipping wandb logging")
        return

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    init_kwargs = {
        "project": cfg.wandb.project,
        "config": config_dict,
        "tags": tags,
    }
    if cfg.wandb.mode:
        init_kwargs["mode"] = cfg.wandb.mode
    wandb.init(**init_kwargs)


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string.
    
    Args:
        image_path: Path to image file
        
    Returns:
        str: Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path to ensure exists
    """
    os.makedirs(path, exist_ok=True)


def get_output_paths(cfg: MainConfig, config_hash: str) -> Dict[str, str]:
    """Get dictionary of output paths based on config.
    
    Args:
        cfg: Configuration object
        config_hash: Hash of training config
        
    Returns:
        Dict[str, str]: Dictionary containing output paths
    """
    return {
        'output_dir': os.path.join(cfg.data.output, "img", config_hash),
        'desc_output_dir': os.path.join(cfg.data.output, "description", config_hash)
    } 

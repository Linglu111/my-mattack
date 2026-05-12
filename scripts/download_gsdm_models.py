"""
GSDM 模型下载脚本

自动下载 GroundingDINO 和 SAM 模型权重到本地缓存。
"""

import os
import sys
from pathlib import Path


def download_grounding_dino():
    print("Downloading GroundingDINO model...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    print(f"GroundingDINO model cached at: {AutoProcessor.from_pretrained(model_id)}")


def download_sam_checkpoint():
    cache_dir = Path.home() / ".cache" / "gsdm_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_dir / "sam_vit_h_4b8939.pth"

    if checkpoint_path.exists():
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"SAM checkpoint already exists: {checkpoint_path} ({size_mb:.1f} MB)")
        return str(checkpoint_path)

    print("Downloading SAM ViT-H checkpoint (2.4GB)...")
    print("This may take several minutes depending on your network speed.")

    import urllib.request

    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(checkpoint_path), _progress)
    print(f"\nDownload complete: {checkpoint_path}")
    return str(checkpoint_path)


def test_models():
    print("\nVerifying models...")

    print("1. Testing GroundingDINO...")
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        "IDEA-Research/grounding-dino-base"
    ).to(device)
    print("   OK - GroundingDINO loaded successfully")

    print("2. Testing SAM...")
    checkpoint = download_sam_checkpoint()
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    print("   OK - SAM loaded successfully")

    print("\nAll models are ready!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_models()
    else:
        print("GSDM Model Downloader")
        print("=" * 50)
        download_grounding_dino()
        download_sam_checkpoint()
        print("\nModels downloaded. Run with --test to verify.")

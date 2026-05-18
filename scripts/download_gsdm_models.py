"""
GSDM 模型下载脚本

自动下载 DINOv2 和 SAM 模型权重到本地缓存。
由于 DINOv2 由 transformers 的 from_pretrained 自动缓存，
此脚本主要处理 SAM 大权重文件的预下载。
"""

import os
import sys
from pathlib import Path


def download_dinov2():
    print("Downloading DINOv2 model (ViT-S/14)...")
    import os
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("Using HF mirror: https://hf-mirror.com")
    from transformers import AutoImageProcessor, AutoModel

    model_id = "facebook/dinov2-small"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    print(f"DINOv2 model cached.")


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
    mirror_urls = [
        "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/sam_vit_h_4b8939.pth",
        "https://hf-mirror.com/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth",
    ]

    def _progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()

    urls_to_try = [url] + mirror_urls
    last_error = None
    for try_url in urls_to_try:
        try:
            print(f"\nTrying: {try_url}")
            urllib.request.urlretrieve(try_url, str(checkpoint_path), _progress)
            print(f"\nDownload complete: {checkpoint_path}")
            return str(checkpoint_path)
        except Exception as e:
            last_error = e
            print(f"\nFailed to download from {try_url}: {e}")
            if checkpoint_path.exists():
                checkpoint_path.unlink()
    raise RuntimeError(f"All download sources failed. Last error: {last_error}")


def test_models():
    print("\nVerifying models...")

    print("1. Testing DINOv2...")
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    model.eval()
    print("   OK - DINOv2 loaded successfully")

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
        print("GSDM Model Downloader (DINOv2 + SAM)")
        print("=" * 50)
        download_dinov2()
        download_sam_checkpoint()
        print("\nModels downloaded. Run with --test to verify.")

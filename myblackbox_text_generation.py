import os
import base64
from PIL import Image
from typing import Dict, Any, List, Tuple
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from config_schema import MainConfig
from openai import OpenAI

from utils import (
    get_api_key,
    hash_training_config,
    setup_wandb,
    ensure_dir,
    encode_image,
    get_output_paths,
)

VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]

# 豆包模型
def setup_doubao(api_key: str):
    return OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )


# 千问模型
def setup_qwen(api_key: str):
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def get_media_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


class ImageDescriptionGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        api_key = get_api_key(model_name)

        if model_name == "doubao":
            self.client = setup_doubao(api_key)
        elif model_name == "qwen":
            self.client = setup_qwen(api_key)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate_description(self, image_path: str) -> str:
        if self.model_name == "doubao":
            return self._generate_doubao(image_path)
        elif self.model_name == "qwen":
            return self._generate_qwen(image_path)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_doubao(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[1].lower()
        media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

        response = self.client.chat.completions.create(
            model="doubao-1-5-vision-pro-32k-250115",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "描述这张图片，不要超过20个字",
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_qwen(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[1].lower()
        media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"

        response = self.client.chat.completions.create(
            model="qwen2.5-vl-72b-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "描述这张图片，不要超过20个字",
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content


def save_descriptions(descriptions: List[Tuple[str, str]], output_file: str):
    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        for filename, desc in descriptions:
            f.write(f"{filename}: {desc}\n")


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    setup_wandb(cfg)

    config_hash = hash_training_config(cfg)
    print(f"Using training output for config hash: {config_hash}")

    paths = get_output_paths(cfg, config_hash)

    if hasattr(cfg.data, 'adv_img_dir') and cfg.data.adv_img_dir:
        paths["output_dir"] = cfg.data.adv_img_dir

    ensure_dir(paths["desc_output_dir"])

    print(f"Output directory (looking for adversarial images): {paths['output_dir']}")
    print(f"Target data path: {cfg.data.tgt_data_path}")
    print(f"Clean data path: {cfg.data.cle_data_path}")

    if not os.path.exists(paths["output_dir"]):
        print(f"ERROR: Output directory does not exist: {paths['output_dir']}")
        return

    try:
        generator = ImageDescriptionGenerator(model_name=cfg.blackbox.model_name)

        tgt_descriptions = []
        adv_descriptions = []
        cle_descriptions = []

        print("Processing images...")

        all_images = []
        for root, _, files in os.walk(paths["output_dir"]):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS):
                    all_images.append(os.path.join(root, file))

        print(f"Found {len(all_images)} image files in {paths['output_dir']}")

        for root, _, files in os.walk(paths["output_dir"]):
            for file in tqdm(files):
                if any(
                    file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS
                ):
                    try:
                        adv_path = os.path.join(root, file)
                        filename_base = os.path.splitext(os.path.basename(adv_path))[0]

                        target_found = False
                        for ext in VALID_IMAGE_EXTENSIONS:
                            tgt_path = os.path.join(
                                cfg.data.tgt_data_path, "1", filename_base + ext
                            )
                            if os.path.exists(tgt_path):
                                target_found = True
                                break

                        clean_found = False
                        for ext in VALID_IMAGE_EXTENSIONS:
                            cle_path = os.path.join(
                                cfg.data.cle_data_path, filename_base + ext
                            )
                            if os.path.exists(cle_path):
                                clean_found = True
                                break

                        if target_found:
                            tgt_desc = generator.generate_description(tgt_path)
                            adv_desc = generator.generate_description(adv_path)

                            tgt_descriptions.append((file, tgt_desc))
                            adv_descriptions.append((file, adv_desc))

                            wandb.log(
                                {
                                    f"descriptions/{file}/target": tgt_desc,
                                    f"descriptions/{file}/adversarial": adv_desc,
                                }
                            )
                        else:
                            attempted_paths = [
                                os.path.join(cfg.data.tgt_data_path, "1", filename_base + ext)
                                for ext in VALID_IMAGE_EXTENSIONS
                            ]
                            print(
                                f"Target image not found for {filename_base}, tried: {attempted_paths[:2]}... skip it."
                            )

                        if clean_found:
                            cle_desc = generator.generate_description(cle_path)
                            cle_descriptions.append((file, cle_desc))

                            wandb.log(
                                {
                                    f"descriptions/{file}/clean": cle_desc,
                                }
                            )

                    except Exception as e:
                        print(f"Error processing {file}: {e}")

        save_descriptions(
            tgt_descriptions,
            os.path.join(
                paths["desc_output_dir"], f"target_{cfg.blackbox.model_name}.txt"
            ),
        )
        save_descriptions(
            adv_descriptions,
            os.path.join(
                paths["desc_output_dir"], f"adversarial_{cfg.blackbox.model_name}.txt"
            ),
        )
        if cle_descriptions:
            save_descriptions(
                cle_descriptions,
                os.path.join(
                    paths["desc_output_dir"], f"clean_{cfg.blackbox.model_name}.txt"
                ),
            )

        print(f"Descriptions saved to {paths['desc_output_dir']}")
        print(f"  Target descriptions: {len(tgt_descriptions)}")
        print(f"  Adversarial descriptions: {len(adv_descriptions)}")
        print(f"  Clean descriptions: {len(cle_descriptions)}")

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        return

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()

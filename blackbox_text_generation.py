import os
import requests
from PIL import Image
from typing import Dict, Any, List, Tuple
import hydra
import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from config_schema import MainConfig
from google import genai
import openai
from openai import OpenAI
import anthropic

from utils import (
    get_api_key,
    hash_training_config,
    setup_wandb,
    ensure_dir,
    encode_image,
    get_output_paths,
)

# 定义有效的图像扩展名
VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]

# 设置Gemini模型
def setup_gemini(api_key: str):
    return genai.Client(api_key=api_key)

# 设置Claude模型
def setup_claude(api_key: str):
    return anthropic.Anthropic(api_key=api_key)

# 设置GPT-4O模型
def setup_gpt4o(api_key: str):
    return OpenAI(
        api_key=api_key,
    )

# 设置豆包模型
def setup_doubao(api_key: str):
    return OpenAI(
        api_key=api_key,
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )


def get_media_type(image_path: str) -> str:
    """根据图像文件名获取正确的媒体类型."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".jpeg"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        raise ValueError(f"Unsupported image extension: {ext}")


class ImageDescriptionGenerator:
    """图像描述生成器."""
    def __init__(self, model_name: str):
        self.model_name = model_name
        # 获取模型的密钥
        api_key = get_api_key(model_name)

        if model_name == "gemini":
            self.client = setup_gemini(api_key)
        elif model_name == "claude":
            self.client = setup_claude(api_key)
        elif model_name == "gpt4o":
            self.client = setup_gpt4o(api_key)
        # 设置豆包模型
        elif model_name == "doubao":
            self.client = setup_doubao(api_key)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def generate_description(self, image_path: str) -> str:
        if self.model_name == "gemini":
            return self._generate_gemini(image_path)
        elif self.model_name == "claude":
            return self._generate_claude(image_path)
        elif self.model_name == "gpt4o":
            return self._generate_gpt4o(image_path)
        elif self.model_name == "doubao":
            return self._generate_doubao(image_path)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    # 生成Gemini模型的图像描述
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gemini(self, image_path: str) -> str:
        image = Image.open(image_path)
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Describe this image, no longer than 25 words.", image],
        )
        return response.text.strip()
    
    # 生成Claude模型的图像描述
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_claude(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image,
                            },
                        },
                    ],
                }
            ],
        )
        return response.content[0].text

    # 生成GPT-4o模型的图像描述
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_gpt4o(self, image_path: str) -> str:
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in one concise sentence, no longer than 20 words.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content

    # 生成豆包模型的图像描述
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _generate_doubao(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            import base64
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        ext = os.path.splitext(image_path)[1].lower()
        media_type = "image/jpeg" if ext in [".jpg", ".jpeg"] else "image/png"
        
        response = self.client.chat.completions.create(
            model="doubao-seed-2-0-lite-260215",
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
    """将图像描述保存到文件中."""
    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        for filename, desc in descriptions:
            f.write(f"{filename}: {desc}\n")


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    # 初始化WandB
    setup_wandb(cfg)

    # 获取配置哈希值
    config_hash = hash_training_config(cfg)
    print(f"Using training output for config hash: {config_hash}")

    # 获取输出路径
    paths = get_output_paths(cfg, config_hash)
    
    # 如果指定了 adv_img_dir，直接使用它作为输出目录
    if hasattr(cfg.data, 'adv_img_dir') and cfg.data.adv_img_dir:
        paths["output_dir"] = cfg.data.adv_img_dir
    
    ensure_dir(paths["desc_output_dir"])
    
    # 调试输出
    print(f"Output directory (looking for adversarial images): {paths['output_dir']}")
    print(f"Target data path: {cfg.data.tgt_data_path}")
    
    # 检查目录是否存在
    if not os.path.exists(paths["output_dir"]):
        print(f"ERROR: Output directory does not exist: {paths['output_dir']}")
        return

    try:
        # 初始化图像描述生成器
        generator = ImageDescriptionGenerator(model_name=cfg.blackbox.model_name)

        # 初始化目标图像描述列表
        tgt_descriptions = []
        # 初始化对抗图像描述列表
        adv_descriptions = []

        # 遍历输出目录，查找对抗图像
        print("Processing images...")
        
        # 调试：列出所有找到的图像文件
        all_images = []
        for root, _, files in os.walk(paths["output_dir"]):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS):
                    all_images.append(os.path.join(root, file))
        
        print(f"Found {len(all_images)} image files in {paths['output_dir']}")
        
        for root, _, files in os.walk(paths["output_dir"]):
            for file in tqdm(files):
                # 检查文件是否为有效图像文件
                if any(
                    file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS
                ):
                    try:
                        # 获取对抗图像路径
                        adv_path = os.path.join(root, file)
                        # 提取文件名（不包含扩展名）用于查找目标图像
                        filename_base = os.path.splitext(os.path.basename(adv_path))[0]

                        # 检查每个有效扩展名，查找目标图像
                        target_found = False
                        for ext in VALID_IMAGE_EXTENSIONS:
                            tgt_path = os.path.join(
                                cfg.data.tgt_data_path, "1", filename_base + ext
                            )
                            if os.path.exists(tgt_path):
                                target_found = True
                                break

                        if target_found:
                            # 生成目标图像描述和对抗图像描述
                            tgt_desc = generator.generate_description(tgt_path)
                            adv_desc = generator.generate_description(adv_path)

                            tgt_descriptions.append((file, tgt_desc))
                            adv_descriptions.append((file, adv_desc))

                            # 记录描述到WandB
                            wandb.log(
                                {
                                    f"descriptions/{file}/target": tgt_desc,
                                    f"descriptions/{file}/adversarial": adv_desc,
                                }
                            )

                        else:
                            # 调试：显示尝试查找的路径
                            attempted_paths = [
                                os.path.join(cfg.data.tgt_data_path, "1", filename_base + ext)
                                for ext in VALID_IMAGE_EXTENSIONS
                            ]
                            print(
                                f"Target image not found for {filename_base}, tried: {attempted_paths[:2]}... skip it."
                            )

                    except Exception as e:
                        print(f"Error processing {file}: {e}")

        # 保存目标图像描述到文件
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

        print(f"Descriptions saved to {paths['desc_output_dir']}")

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
        return

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()

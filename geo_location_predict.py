"""Query a vision-language model for one GPS prediction per image.

The API calling code is split by provider so Gemini, OpenAI-compatible models,
and Qwen can be configured independently. This script only performs inference;
run geo_location_score.py afterwards to compute geolocation accuracy.
"""

import argparse
import base64
import json
import mimetypes
import re
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm


# Fill these locally or pass --api-key / --base-url from the command line.
GEMINI_API_KEY = "sk-X6XPX7XI19twx71ICHHosQbc9CfTZovAf6iXyNMTDR54oMFT"
GEMINI_BASE_URL = "https://api.openai-proxy.org/google"
GEMINI_MODEL = "gemini-2.5-pro"

OPENAI_API_KEY = "sk-X6XPX7XI19twx71ICHHosQbc9CfTZovAf6iXyNMTDR54oMFT"
OPENAI_BASE_URL = "https://api.openai-proxy.org/v1"
OPENAI_MODEL = "gpt-4.1"

QWEN_API_KEY = ""
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_MODEL = "qwen-vl-max"

PROMPT = """You are solving a worldwide image geolocation benchmark.
Estimate where the image was taken using only visible geographic clues such as road layout, architecture, vegetation, terrain, signs, lane markings, vehicles, and climate.
Return exactly one best-guess WGS84 GPS coordinate.
Output only valid JSON in this schema:
{"latitude": 12.345678, "longitude": 98.765432}
The latitude must be in [-90, 90] and the longitude must be in [-180, 180].
Do not output a single number, place name, explanation, markdown, or extra text."""
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one GPS coordinate for each image.")
    parser.add_argument(
        "--image-dir",
        default="LAT/img/d0ced9cdda9d030480af7288efe3525c",
        help="Image directory, searched recursively.",
    )
    parser.add_argument(
        "--output",
        default="LAT/geolocation_predictions/predictions.jsonl",
        help="JSONL output path.",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "qwen"],
        default="gemini",
        help="API provider/client type.",
    )
    parser.add_argument("--api-key", default=None, help="API key. Overrides provider default.")
    parser.add_argument("--base-url", default=None, help="Provider base URL. Overrides provider default.")
    parser.add_argument("--model", default=None, help="Model name. Overrides provider default.")
    parser.add_argument("--prompt", default=PROMPT, help="Prompt sent with each image.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N images.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls.")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum output tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip existing rows.")
    return parser.parse_args()


def provider_defaults(provider: str) -> Tuple[str, str, str]:
    if provider == "gemini":
        return GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL
    if provider == "openai":
        return OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    if provider == "qwen":
        return QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL
    raise ValueError(f"Unsupported provider: {provider}")


def iter_images(image_dir: str) -> Iterable[Path]:
    root = Path(image_dir)
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def load_done_images(output_path: Path) -> set:
    done = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("image_path"):
                done.add(row["image_path"])
    return done


def valid_coordinate(latitude: float, longitude: float) -> bool:
    return -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0


def parse_coordinate(text: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Parse latitude/longitude from simple Chinese/English coordinate text."""
    cleaned = text.strip()

    json_match = re.search(r"\{.*?\}", cleaned, flags=re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            lat = data.get("latitude", data.get("lat", data.get("纬度")))
            lon = data.get("longitude", data.get("lon", data.get("lng", data.get("经度"))))
            if lat is not None and lon is not None:
                latitude = float(lat)
                longitude = float(lon)
                if valid_coordinate(latitude, longitude):
                    return latitude, longitude, None
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    lat_match = re.search(
        r"(?:纬度|latitude|lat)\s*[=:：]?\s*([-+]?\d+(?:\.\d+)?)",
        cleaned,
        flags=re.IGNORECASE,
    )
    lon_match = re.search(
        r"(?:经度|longitude|lon|lng)\s*[=:：]?\s*([-+]?\d+(?:\.\d+)?)",
        cleaned,
        flags=re.IGNORECASE,
    )
    if lat_match and lon_match:
        latitude = float(lat_match.group(1))
        longitude = float(lon_match.group(1))
        if valid_coordinate(latitude, longitude):
            return latitude, longitude, None

    for first, second in re.findall(
        r"([-+]?\d+(?:\.\d+)?)\s*[,，]\s*([-+]?\d+(?:\.\d+)?)",
        cleaned,
    ):
        latitude = float(first)
        longitude = float(second)
        if valid_coordinate(latitude, longitude):
            return latitude, longitude, None

    return None, None, "Could not parse latitude/longitude."


def encode_image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    with image_path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def read_image_bytes(image_path: Path) -> Tuple[bytes, str]:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
    with image_path.open("rb") as handle:
        return handle.read(), mime_type


class BaseGeoClient:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        max_tokens: int,
        temperature: float,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    def predict(self, image_path: Path, prompt: str) -> str:
        raise NotImplementedError


class GeminiGeoClient(BaseGeoClient):
    """Gemini client using the latest google-genai SDK style."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Install or upgrade the new Google Gen AI SDK to use --provider gemini: "
                "pip install -U google-genai. This provider uses `from google import genai`, "
                "not the old `google.generativeai` SDK."
            ) from exc

        self._genai = genai
        self._types = None
        try:
            from google.genai import types

            self._types = types
        except ImportError:
            self._types = None

        self.client = genai.Client(
            api_key=self.api_key,
            vertexai=True,
            http_options={"base_url": self.base_url},
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
    def predict(self, image_path: Path, prompt: str) -> str:
        config = None
        if self._types is not None:
            config = self._types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            )

        image_bytes, mime_type = read_image_bytes(image_path)
        if self._types is not None:
            image_part = self._types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        else:
            with Image.open(image_path) as image:
                image_part = image.copy()

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, image_part],
            config=config,
        )
        return getattr(response, "text", "").strip()


class OpenAICompatibleGeoClient(BaseGeoClient):
    """OpenAI-compatible vision chat client for OpenAI, Qwen, or proxy providers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use --provider openai or --provider qwen.") from exc

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)
    def predict(self, image_path: Path, prompt: str) -> str:
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_image_data_url(image_path)},
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as exc:
            if "response_format" not in str(exc):
                raise
            request.pop("response_format")
            response = self.client.chat.completions.create(**request)
        return response.choices[0].message.content.strip()


class QwenGeoClient(OpenAICompatibleGeoClient):
    """Qwen vision models through DashScope/OpenAI-compatible endpoints."""


def build_client(args: argparse.Namespace) -> BaseGeoClient:
    default_key, default_base_url, default_model = provider_defaults(args.provider)
    api_key = args.api_key if args.api_key is not None else default_key
    base_url = args.base_url if args.base_url is not None else default_base_url
    model = args.model if args.model is not None else default_model

    if not api_key:
        raise ValueError(
            f"API key is empty for provider={args.provider}. "
            "Fill the provider key constant or pass --api-key."
        )

    kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "timeout": args.timeout,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.provider == "gemini":
        return GeminiGeoClient(**kwargs)
    if args.provider == "openai":
        return OpenAICompatibleGeoClient(**kwargs)
    if args.provider == "qwen":
        return QwenGeoClient(**kwargs)
    raise ValueError(f"Unsupported provider: {args.provider}")


def main() -> None:
    args = parse_args()
    client = build_client(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(args.image_dir))
    if args.limit is not None:
        images = images[: args.limit]
    done_images = set() if args.no_resume else load_done_images(output_path)

    skipped = 0
    with output_path.open("a", encoding="utf-8") as handle:
        pbar = tqdm(images, desc=f"Querying {args.provider}")
        for image_path in pbar:
            image_key = str(image_path)
            if image_key in done_images:
                skipped += 1
                pbar.set_postfix_str(f"skip {image_path.name} (total skipped: {skipped})")
                continue

            pbar.set_postfix_str(f"query {image_path.name}")
            try:
                raw_response = client.predict(image_path, args.prompt)
                latitude, longitude, parse_error = parse_coordinate(raw_response)
                model_error = None
            except Exception as exc:
                raw_response = ""
                latitude, longitude = None, None
                parse_error = None
                model_error = f"API query failed after retries: {exc}"

            row = {
                "image_path": image_key,
                "image_name": image_path.name,
                "image_stem": image_path.stem,
                "provider": args.provider,
                "model": client.model,
                "base_url": client.base_url,
                "prompt": args.prompt,
                "raw_response": raw_response,
                "pred_latitude": latitude,
                "pred_longitude": longitude,
                "parse_error": parse_error,
                "model_error": model_error,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            done_images.add(image_key)

            if args.sleep > 0:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()

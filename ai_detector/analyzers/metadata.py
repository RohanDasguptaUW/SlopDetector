"""Metadata analyzer — EXIF and PNG chunk inspection."""

from PIL import Image
from PIL.ExifTags import TAGS

from .base import AnalysisResult, BaseAnalyzer

_AI_SOFTWARE_STRINGS = [
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dall·e",
    "firefly",
    "comfyui",
    "automatic1111",
    "invokeai",
    "novelai",
    "dreamstudio",
    "adobe firefly",
    "imagen",
    "bing image creator",
    "ai image",
    "generative",
    "diffusion",
]

_EXIF_TAG_IDS = {v: k for k, v in TAGS.items()}


def _get_exif_tag(exif_data: dict, tag_name: str):
    tag_id = _EXIF_TAG_IDS.get(tag_name)
    if tag_id is None:
        return None
    return exif_data.get(tag_id)


def has_camera_exif(image_path: str) -> bool:
    """Return True if the image has a camera Make or Model EXIF tag."""
    try:
        img = Image.open(image_path)
        raw = img._getexif()  # type: ignore[attr-defined]
        if not raw:
            return False
        exif_data = raw
    except Exception:
        return False
    make = _get_exif_tag(exif_data, "Make")
    model_tag = _get_exif_tag(exif_data, "Model")
    return bool(make or model_tag)


class MetadataAnalyzer(BaseAnalyzer):
    name = "metadata"

    def analyze(self, image_path: str) -> AnalysisResult:
        img = Image.open(image_path)

        score = 0.0
        indicators: list[str] = []

        # --- EXIF ---
        exif_data: dict = {}
        try:
            raw = img._getexif()  # type: ignore[attr-defined]
            if raw:
                exif_data = raw
        except (AttributeError, Exception):
            pass

        software = _get_exif_tag(exif_data, "Software")
        if software:
            low = str(software).lower()
            for sig in _AI_SOFTWARE_STRINGS:
                if sig in low:
                    score += 80.0
                    indicators.append(f"EXIF Software tag identifies AI generator: '{software}'")
                    break
        else:
            score += 10.0
            indicators.append("No EXIF Software tag present")

        make = _get_exif_tag(exif_data, "Make")
        model_tag = _get_exif_tag(exif_data, "Model")
        if not make and not model_tag:
            score += 15.0
            indicators.append("No camera Make/Model in EXIF — not from a physical camera")

        datetime_orig = _get_exif_tag(exif_data, "DateTimeOriginal")
        if not datetime_orig:
            score += 10.0
            indicators.append("No DateTimeOriginal in EXIF")

        # --- PNG chunks (img.info) ---
        png_info = img.info or {}
        for key, value in png_info.items():
            val_str = str(value).lower()
            key_str = str(key).lower()
            for sig in _AI_SOFTWARE_STRINGS:
                if sig in val_str or sig in key_str:
                    score += 70.0
                    indicators.append(f"PNG chunk '{key}' contains AI generator reference")
                    break

        # Check for "parameters" chunk used by Stable Diffusion / ComfyUI
        if "parameters" in png_info:
            score += 60.0
            indicators.append("PNG 'parameters' chunk found — typical Stable Diffusion output")
        if "prompt" in png_info or "workflow" in png_info:
            score += 55.0
            indicators.append("PNG contains AI workflow/prompt metadata")
        if "comment" in png_info:
            comment_low = str(png_info["comment"]).lower()
            for sig in _AI_SOFTWARE_STRINGS:
                if sig in comment_low:
                    score += 65.0
                    indicators.append("PNG comment references AI generator")
                    break

        if not indicators:
            indicators.append("No suspicious metadata found")

        ai_percentage = float(min(score, 100.0))

        # Confidence: metadata is deterministic — high when we have a match, low otherwise
        if ai_percentage > 50:
            confidence = 0.95
        elif ai_percentage > 15:
            confidence = 0.60
        else:
            confidence = 0.40

        return AnalysisResult(
            analyzer=self.name,
            ai_percentage=ai_percentage,
            confidence=confidence,
            indicators=indicators,
            heatmap=None,
        )

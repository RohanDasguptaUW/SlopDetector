"""Gemini AI analyzer — single API call with 3×3 grid scores."""

import base64
import io
import json
import os
import re

import numpy as np
from PIL import Image

from .base import AnalysisResult, BaseAnalyzer

_SYSTEM_PROMPT = """\
You are an expert forensic image analyst specializing in detecting AI-generated images.
Analyze the provided image and respond ONLY with a valid JSON object — no prose, no markdown fences.

Required JSON format:
{
  "ai_percentage": <0-100 float, overall probability the image is AI-generated>,
  "confidence": <0-1 float, your confidence in this estimate>,
  "grid_scores": [[<row0col0>, <row0col1>, <row0col2>],
                  [<row1col0>, <row1col1>, <row1col2>],
                  [<row2col0>, <row2col1>, <row2col2>]],
  "indicators": [<list of short strings describing observed signals>]
}

grid_scores: Each cell is 0–10 (0=likely real, 10=likely AI) for that spatial region of the image.
Rows are top-to-bottom, columns are left-to-right.

Signals to examine:
- Unnatural texture smoothness or excessive detail regularity
- Anatomical errors (extra/missing fingers, merged limbs, odd faces)
- Lighting/shadow inconsistencies
- Background incoherence or repeating patterns
- GAN checkerboard artefacts in high-frequency regions
- Watermarks or metadata-like text patterns
- Overly perfect skin, hair, or fabric textures
- Surreal or physically impossible elements
"""


def _bicubic_interpolate_grid(grid_3x3: list[list[float]], target_h: int, target_w: int) -> np.ndarray:
    """Bicubic-interpolate a 3×3 score grid to (target_h, target_w)."""
    arr = np.array(grid_3x3, dtype=np.float32)
    pil_grid = Image.fromarray((arr / 10.0 * 255).astype(np.uint8), mode="L")
    upscaled = pil_grid.resize((target_w, target_h), resample=Image.BICUBIC)
    return np.array(upscaled, dtype=np.float32) / 255.0


class GeminiAnalyzer(BaseAnalyzer):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model

    def analyze(self, image_path: str) -> AnalysisResult:
        try:
            import google.generativeai as genai  # type: ignore[import]
        except ImportError:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["google-generativeai package not installed — Gemini analysis skipped"],
            )

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["GOOGLE_API_KEY not set — Gemini analysis skipped"],
            )

        img = Image.open(image_path)
        H, W = img.size[1], img.size[0]

        # Downscale to ≤1MP for speed/cost
        max_pixels = 1_000_000
        if H * W > max_pixels:
            scale = (max_pixels / (H * W)) ** 0.5
            img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)

        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=_SYSTEM_PROMPT,
        )

        try:
            response = gemini_model.generate_content(
                [img.convert("RGB"), "Analyse this image and return only the JSON object as specified."]
            )
        except Exception as exc:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=[f"Gemini API error: {exc}"],
            )

        raw_text = response.text or ""

        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["Gemini returned non-JSON response"],
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["Failed to parse Gemini JSON response"],
            )

        ai_percentage = float(np.clip(data.get("ai_percentage", 50.0), 0, 100))
        confidence = float(np.clip(data.get("confidence", 0.5), 0, 1))
        indicators: list[str] = data.get("indicators", [])

        grid_scores = data.get("grid_scores")
        heatmap: np.ndarray | None = None
        if grid_scores and len(grid_scores) == 3 and all(len(r) == 3 for r in grid_scores):
            heatmap = _bicubic_interpolate_grid(grid_scores, H, W)

        return AnalysisResult(
            analyzer=self.name,
            ai_percentage=ai_percentage,
            confidence=confidence,
            indicators=indicators if indicators else ["No specific indicators identified"],
            heatmap=heatmap,
        )

"""Claude AI analyzer — single API call with 3×3 grid scores."""

import base64
import json
import re
import os
from pathlib import Path

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
    arr = np.array(grid_3x3, dtype=np.float32)  # shape (3, 3)
    pil_grid = Image.fromarray((arr / 10.0 * 255).astype(np.uint8), mode="L")
    upscaled = pil_grid.resize((target_w, target_h), resample=Image.BICUBIC)
    result = np.array(upscaled, dtype=np.float32) / 255.0
    return result


class ClaudeAnalyzer(BaseAnalyzer):
    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model

    def analyze(self, image_path: str) -> AnalysisResult:
        try:
            import anthropic
        except ImportError:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["anthropic package not installed — Claude analysis skipped"],
                heatmap=None,
            )

        img = Image.open(image_path)
        H, W = img.size[1], img.size[0]

        # Encode image as base64 JPEG (≤1MP for speed)
        max_pixels = 1_000_000
        if H * W > max_pixels:
            scale = (max_pixels / (H * W)) ** 0.5
            img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)

        import io
        buf = io.BytesIO()
        rgb = img.convert("RGB")
        rgb.save(buf, format="JPEG", quality=85)
        b64_data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

        client = anthropic.Anthropic()

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": b64_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": "Analyse this image and return only the JSON object as specified.",
                            },
                        ],
                    }
                ],
            )
        except Exception as exc:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=[f"Claude API error: {exc}"],
                heatmap=None,
            )

        raw_text = ""
        for block in response.content:
            if block.type == "text":
                raw_text = block.text
                break

        # Extract JSON even if Claude added surrounding text
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["Claude returned non-JSON response"],
                heatmap=None,
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["Failed to parse Claude JSON response"],
                heatmap=None,
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

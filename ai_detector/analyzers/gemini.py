"""Gemini AI analyzer — single API call with 3×3 grid scores."""

import io
import json
import os
import re

import numpy as np
from PIL import Image

from .base import AnalysisResult, BaseAnalyzer

_SYSTEM_PROMPT = """\
You are an adversarial forensic analyst whose sole job is to expose AI-generated images.
Treat every image as a suspect. Your default assumption is that the image is AI-generated,
and you must find strong, specific evidence of authenticity before scoring it below 40%.
Respond ONLY with a valid JSON object — no prose, no markdown fences.

Required JSON format:
{
  "ai_percentage": <0-100 float, probability the image is AI-generated>,
  "confidence": <0-1 float, your confidence in this estimate>,
  "grid_scores": [[<row0col0>, <row0col1>, <row0col2>],
                  [<row1col0>, <row1col1>, <row1col2>],
                  [<row2col0>, <row2col1>, <row2col2>]],
  "indicators": [<list of short strings, each naming a specific signal found>]
}

grid_scores: Each cell is 0–10 (0=likely real, 10=likely AI) for that spatial region.
Rows are top-to-bottom, columns are left-to-right.

FORENSIC CHECKS — examine each one aggressively:

SKIN TEXTURE
Zoom in mentally on any visible skin. Real skin has pores, fine hairs, uneven pigmentation,
and micro-texture that is chaotic and asymmetric. AI skin is characteristically over-smoothed,
with procedural noise that looks like a texture map — regular, directionless, and too clean.
Flag any skin that lacks genuine micro-imperfection.

BOKEH / DEPTH-OF-FIELD
Real optical bokeh has: elliptical or cat-eye distortion toward frame edges, chromatic
aberration fringing, onion-ring structures inside blur circles, and blur that transitions
abruptly at depth discontinuities. AI bokeh is computationally generated — it is round,
spectrally clean, transitions smoothly everywhere, and looks like a Gaussian blur mask.
Flag suspiciously perfect or uniform background blur.

FACIAL SYMMETRY
Human faces are never truly symmetric — one eye is slightly lower, one ear protrudes more,
the nose deviates slightly. AI-generated faces tend toward hyper-symmetry that no real person
has. Check the horizontal midline carefully. Flag any face where both halves feel like
mirror images of each other.

FABRIC AND MATERIAL TEXTURE
Real cloth has: loose threads, irregular weave density, localised creases that don't repeat,
areas of wear, and lint. AI fabric textures tile or repeat at a local level — look for any
patch of cloth where the texture pattern appears elsewhere in the same garment. Flag
periodicity in material texture.

EDGE HALOS
Diffusion models characteristically produce a soft luminance halo around object edges —
a slight brightening or blurring at the boundary between subject and background that no
real camera produces. Examine silhouettes, hair edges, and clothing outlines for this
synthetic glow or feathering.

LIGHTING
Real light sources are imperfect: they produce hard falloff, colour temperature shifts
across a scene, secondary bounce fills that are colour-tinted, and shadows with slightly
soft but directionally consistent edges. AI lighting is render-like — evenly distributed,
specular highlights are placed where they look good rather than where physics demands,
and shadows may be directionless or missing entirely. Flag lighting that looks like
a three-point studio setup with no physical light source visible.

ABSENCE OF MICRO-IMPERFECTIONS
Real photographs always contain: dust particles on surfaces, slight skin unevenness or
blemishes, at least one stray hair out of place, smudges on glasses, asymmetric clothing
drape, and small environmental debris. The complete absence of any such imperfections is
itself a strong AI signal. Flag images that are too clean to be real.

SCORING DISCIPLINE
You are a skeptic. Do not score below 40% unless you can cite multiple specific,
concrete authenticity signals — visible sensor noise, identifiable optical aberrations,
genuine anatomical asymmetry, real fabric imperfection, etc. Ambiguity counts against
authenticity, not against AI. If you are uncertain, score high.
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
            from google import genai  # type: ignore[import]
            from google.genai import types  # type: ignore[import]
        except ImportError:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["google-genai package not installed — Gemini analysis skipped"],
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

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

        client = genai.Client(api_key=api_key)

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=[image_part, "Analyse this image and return only the JSON object as specified."],
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                ),
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

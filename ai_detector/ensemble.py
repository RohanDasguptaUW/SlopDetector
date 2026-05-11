"""Ensemble combiner for analyzer results."""

import numpy as np
from typing import Optional

from .analyzers.base import AnalysisResult

_DEFAULT_WEIGHTS: dict[str, float] = {
    "ela": 0.35,
    "noise": 0.25,
    "gemini": 0.20,
    "metadata": 0.15,
    "spectral": 0.05,
}


def combine(results: list[AnalysisResult]) -> dict:
    """Combine analyzer results into a weighted ensemble summary."""
    if not results:
        return {
            "ai_percentage": 50.0,
            "confidence": 0.0,
            "verdict": "Unknown",
            "weights_used": {},
            "per_analyzer": {},
        }

    # Build weight map; redistribute if some analyzers are missing/failed
    available = {r.analyzer for r in results}
    raw_weights = {name: w for name, w in _DEFAULT_WEIGHTS.items() if name in available}

    # Normalise weights to sum to 1
    total_w = sum(raw_weights.values())
    if total_w == 0:
        weights: dict[str, float] = {r.analyzer: 1.0 / len(results) for r in results}
    else:
        weights = {name: w / total_w for name, w in raw_weights.items()}
        # Any result whose analyzer has no default weight gets equal share of remainder
        unweighted = [r for r in results if r.analyzer not in weights]
        if unweighted:
            extra = (1.0 - sum(weights.values())) / len(unweighted)
            for r in unweighted:
                weights[r.analyzer] = extra

    # Weighted average of ai_percentage and confidence
    ai_sum = 0.0
    conf_sum = 0.0
    w_sum = 0.0
    for r in results:
        w = weights.get(r.analyzer, 0.0)
        ai_sum += w * r.ai_percentage
        conf_sum += w * r.confidence
        w_sum += w

    ai_percentage = ai_sum / w_sum if w_sum > 0 else 50.0
    confidence = conf_sum / w_sum if w_sum > 0 else 0.0

    # Combine spatial heatmaps as weighted average
    heatmap: Optional[np.ndarray] = None
    heatmap_w_sum = 0.0
    for r in results:
        if r.heatmap is not None:
            w = weights.get(r.analyzer, 0.0)
            if heatmap is None:
                heatmap = w * r.heatmap.astype(np.float32)
            else:
                # Resize to match if needed
                if heatmap.shape != r.heatmap.shape:
                    from PIL import Image
                    pil_h = Image.fromarray((r.heatmap * 255).astype(np.uint8))
                    pil_h = pil_h.resize((heatmap.shape[1], heatmap.shape[0]), Image.BILINEAR)
                    scaled = np.array(pil_h, dtype=np.float32) / 255.0
                    heatmap += w * scaled
                else:
                    heatmap += w * r.heatmap.astype(np.float32)
            heatmap_w_sum += w

    if heatmap is not None and heatmap_w_sum > 0:
        heatmap /= heatmap_w_sum

    verdict = _verdict(ai_percentage, confidence)

    return {
        "ai_percentage": round(ai_percentage, 2),
        "confidence": round(confidence, 3),
        "verdict": verdict,
        "heatmap": heatmap,
        "weights_used": {k: round(v, 4) for k, v in weights.items()},
        "per_analyzer": {r.analyzer: r.to_dict() for r in results},
    }


def _verdict(ai_percentage: float, confidence: float) -> str:
    if confidence < 0.3:
        return "Inconclusive"
    if ai_percentage >= 75:
        return "Likely AI-Generated"
    if ai_percentage >= 45:
        return "Possibly AI-Generated"
    if ai_percentage >= 20:
        return "Probably Real"
    return "Likely Real"

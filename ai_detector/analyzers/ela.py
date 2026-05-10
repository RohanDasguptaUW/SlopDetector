"""Error Level Analysis (ELA) analyzer."""

import io
import numpy as np
from PIL import Image

from .base import AnalysisResult, BaseAnalyzer


class ELAAnalyzer(BaseAnalyzer):
    name = "ela"

    def analyze(self, image_path: str) -> AnalysisResult:
        original = Image.open(image_path).convert("RGB")

        buf = io.BytesIO()
        original.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        orig_arr = np.array(original, dtype=np.float32)
        comp_arr = np.array(recompressed, dtype=np.float32)

        diff = np.abs(orig_arr - comp_arr) * 12.0
        diff = np.clip(diff, 0, 255)

        # Luminance-weighted grayscale
        weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        ela_gray = (diff * weights).sum(axis=2)

        mean_amp = float(ela_gray.mean())
        std_amp = float(ela_gray.std())
        cv = std_amp / (mean_amp + 1e-6)

        # Normalise for heatmap
        heatmap = ela_gray / (ela_gray.max() + 1e-6)
        heatmap = heatmap.astype(np.float32)

        # Low CV → suspiciously uniform → higher AI score
        # High mean → more difference from JPEG recompression → more real
        # Score: blend of inverse-CV and inverse mean
        cv_score = max(0.0, 1.0 - cv)          # 0=natural, 1=uniform (AI-like)
        amp_score = max(0.0, 1.0 - mean_amp / 30.0)  # low amplitude also suspicious

        raw_score = 0.5 * cv_score + 0.5 * amp_score
        ai_percentage = float(np.clip(raw_score * 100, 0, 100))

        # Confidence higher when signal is clear
        confidence = float(np.clip(0.4 + 0.6 * abs(cv_score - 0.5) * 2, 0, 1))

        indicators: list[str] = []
        if cv < 0.5:
            indicators.append(f"Suspiciously uniform ELA response (CV={cv:.3f})")
        if mean_amp < 5.0:
            indicators.append(f"Very low ELA amplitude ({mean_amp:.2f}) — minimal JPEG artefacts")
        if mean_amp > 20.0:
            indicators.append(f"High ELA amplitude ({mean_amp:.2f}) — natural JPEG noise present")

        return AnalysisResult(
            analyzer=self.name,
            ai_percentage=ai_percentage,
            confidence=confidence,
            indicators=indicators,
            heatmap=heatmap,
        )

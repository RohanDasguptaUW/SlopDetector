"""Noise statistics analyzer — shot noise model and over-smoothness detection."""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

from .base import AnalysisResult, BaseAnalyzer


class NoiseAnalyzer(BaseAnalyzer):
    name = "noise"

    def analyze(self, image_path: str) -> AnalysisResult:
        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img, dtype=np.float32)
        H, W = img_arr.shape[:2]

        # Green channel — least affected by demosaicing and color processing
        green = img_arr[:, :, 1]

        # ── Signal 1: Shot noise correlation ────────────────────────────────
        # Real cameras obey Poisson photon statistics: noise variance scales with
        # signal level. AI image generators have no sensor, so this physical
        # relationship is absent — correlation is near zero or inverted.
        smooth = gaussian_filter(green, sigma=2.0)
        noise_residual = green - smooth

        ps = max(16, min(H, W) // 16)
        patch_means: list[float] = []
        patch_vars: list[float] = []
        for y in range(0, H - ps, ps):
            for x in range(0, W - ps, ps):
                patch_means.append(float(smooth[y:y+ps, x:x+ps].mean()))
                patch_vars.append(float(np.var(noise_residual[y:y+ps, x:x+ps])))

        corr = 0.0
        if len(patch_means) > 10:
            pm = np.array(patch_means)
            pv = np.array(patch_vars)
            if pm.std() > 0.5 and pv.std() > 0:
                corr = float(np.corrcoef(pm, pv)[0, 1])

        # Positive corr → Poisson-consistent → real photo; low/negative → AI
        shot_noise_score = float(np.clip(0.5 - corr * 0.5, 0.0, 1.0))

        # ── Signal 2: Smooth-region over-smoothness ──────────────────────────
        # Real photos retain micro-texture (pores, grain, sensor noise) even in
        # apparently smooth areas. AI diffusion models produce regions that are
        # unnaturally clean at the pixel level.
        sobel_x = np.gradient(green, axis=1)
        sobel_y = np.gradient(green, axis=0)
        gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        grad_threshold = float(np.percentile(gradient_mag, 40))
        block = max(8, min(H, W) // 32)
        smooth_region_vars: list[float] = []
        for y in range(0, H - block, block):
            for x in range(0, W - block, block):
                if float(gradient_mag[y:y+block, x:x+block].mean()) <= grad_threshold:
                    smooth_region_vars.append(float(np.var(green[y:y+block, x:x+block])))

        oversmooth_score = 0.5
        median_var = -1.0
        if len(smooth_region_vars) > 5:
            median_var = float(np.median(smooth_region_vars))
            # Real photo noise floor in smooth patches: ~10–80 depending on ISO.
            # AI-generated smooth patches: typically < 5.
            oversmooth_score = float(np.clip(1.0 - median_var / 40.0, 0.0, 1.0))

        # ── Heatmap: inverted local variance (low variance = AI-like = hot) ──
        win = max(5, min(H, W) // 64)
        mean_map = uniform_filter(green, size=win)
        sq_map = uniform_filter(green ** 2, size=win)
        var_map = np.maximum(sq_map - mean_map ** 2, 0.0)
        var95 = float(np.percentile(var_map, 95)) + 1e-6
        heatmap = (1.0 - np.clip(var_map / var95, 0.0, 1.0)).astype(np.float32)

        # ── Score ────────────────────────────────────────────────────────────
        # Shot-noise Poisson correlation is destroyed by JPEG compression,
        # sharpening, and any post-processing, so it fires on real photos almost
        # as often as on AI images — useless as a standalone signal. Exclude it
        # from the score and only incorporate it when over-smoothness independently
        # indicates AI (both signals must agree before shot-noise contributes).
        shot_noise_corroborates = oversmooth_score > 0.6
        if shot_noise_corroborates:
            raw_score = 0.25 * shot_noise_score + 0.75 * oversmooth_score
        else:
            raw_score = oversmooth_score
        ai_percentage = float(np.clip(raw_score * 100, 0, 100))
        confidence = float(np.clip(0.35 + 0.50 * oversmooth_score, 0, 0.80))

        indicators: list[str] = []
        if corr < 0.15 and shot_noise_corroborates:
            indicators.append(
                f"Shot noise model absent (signal–noise corr={corr:.2f}) — no camera sensor signature"
            )
        if 0 <= median_var < 10.0:
            indicators.append(
                f"Smooth regions unnaturally clean (local var={median_var:.1f}) — AI over-smoothing"
            )
        if corr > 0.5:
            indicators.append(f"Shot noise model present (corr={corr:.2f}) — consistent with real camera")
        if median_var >= 10.0:
            indicators.append(f"Natural micro-texture in smooth regions (local var={median_var:.1f})")
        if not indicators:
            indicators.append("Noise statistics inconclusive")

        return AnalysisResult(
            analyzer=self.name,
            ai_percentage=ai_percentage,
            confidence=confidence,
            indicators=indicators,
            heatmap=heatmap,
        )

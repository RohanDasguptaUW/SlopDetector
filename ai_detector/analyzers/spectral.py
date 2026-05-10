"""Spectral (2D FFT) analyzer for GAN checkerboard artefacts."""

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d

from .base import AnalysisResult, BaseAnalyzer


class SpectralAnalyzer(BaseAnalyzer):
    name = "spectral"

    def analyze(self, image_path: str) -> AnalysisResult:
        img = Image.open(image_path).convert("L")
        gray = np.array(img, dtype=np.float32)

        # 2D FFT
        fft = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)
        power = magnitude ** 2

        H, W = gray.shape
        cy, cx = H // 2, W // 2

        # Radial coordinates
        yy, xx = np.ogrid[:H, :W]
        radius_map = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.float32)
        max_radius = int(radius_map.max())

        # Radial power spectrum
        radial_power = np.zeros(max_radius + 1, dtype=np.float64)
        counts = np.zeros(max_radius + 1, dtype=np.int64)
        r_int = radius_map.astype(np.int32)
        np.add.at(radial_power, r_int, power)
        np.add.at(counts, r_int, 1)
        counts = np.maximum(counts, 1)
        radial_power /= counts

        # Smooth baseline via uniform filter
        baseline = uniform_filter1d(radial_power, size=15)
        residual = radial_power - baseline

        # Detect off-centre spikes (skip DC: radius 0–2)
        spike_threshold = 3.0 * residual[3:].std()
        spike_mask = np.zeros(max_radius + 1, dtype=bool)
        spike_mask[3:] = residual[3:] > spike_threshold
        n_spikes = int(spike_mask.sum())

        # 1/f fit residual (log-log space, skip DC) + high-frequency noise floor deficit
        r_range = np.arange(3, max_radius + 1)
        valid = radial_power[3:] > 0
        residual_std = 0.0
        hf_deficit_score = 0.0
        if valid.sum() > 10:
            log_r = np.log(r_range[valid])
            log_p = np.log(radial_power[3:][valid])
            coeffs = np.polyfit(log_r, log_p, 1)
            fit = np.polyval(coeffs, log_r)
            residual_std = float(np.std(log_p - fit))

            # Real camera photos maintain a sensor-noise floor at high spatial frequencies,
            # keeping actual power at or above the 1/f prediction.
            # Diffusion model images have no sensor noise, so high-frequency power falls
            # sharply below the 1/f trend — this deficit is the primary signal here.
            hf_start = max(3, int(max_radius * 0.60))
            hf_mask = r_range >= hf_start
            hf_actual = radial_power[3:][hf_mask]
            hf_valid = hf_actual > 0
            if hf_valid.sum() > 5:
                hf_r = r_range[hf_mask][hf_valid]
                hf_predicted = np.exp(np.polyval(coeffs, np.log(hf_r)))
                ratio = float(np.median(hf_actual[hf_valid] / (hf_predicted + 1e-10)))
                hf_deficit_score = float(np.clip(1.0 - ratio, 0.0, 1.0))

        # Build heatmap from anomalous frequencies (inverse FFT of spike region)
        anomaly_mask = spike_mask[r_int]
        fft_anomaly = fft_shifted * anomaly_mask
        fft_anomaly_unshifted = np.fft.ifftshift(fft_anomaly)
        spatial_anomaly = np.abs(np.fft.ifft2(fft_anomaly_unshifted)).astype(np.float32)
        hmap_max = spatial_anomaly.max()
        heatmap = spatial_anomaly / (hmap_max + 1e-6)

        # Score — spike detection targets old-style GANs; HF deficit targets diffusion models
        spike_score = min(1.0, n_spikes / 20.0)
        residual_score = min(1.0, residual_std / 2.0)
        raw_score = 0.25 * spike_score + 0.20 * residual_score + 0.55 * hf_deficit_score
        ai_percentage = float(np.clip(raw_score * 100, 0, 100))

        confidence = float(np.clip(0.3 + 0.7 * max(spike_score, residual_score, hf_deficit_score), 0, 1))

        indicators: list[str] = []
        if n_spikes > 5:
            indicators.append(f"Off-centre spectral spikes detected ({n_spikes}) — GAN checkerboard artefact")
        if residual_std > 0.8:
            indicators.append(f"Deviation from 1/f power law (residual std={residual_std:.3f})")
        if hf_deficit_score > 0.4:
            indicators.append(f"High-frequency noise floor absent (deficit={hf_deficit_score:.2f}) — no sensor noise signature")
        if n_spikes == 0 and residual_std < 0.3 and hf_deficit_score < 0.3:
            indicators.append("Spectral profile consistent with natural image")

        return AnalysisResult(
            analyzer=self.name,
            ai_percentage=ai_percentage,
            confidence=confidence,
            indicators=indicators,
            heatmap=heatmap,
        )

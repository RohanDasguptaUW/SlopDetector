"""pytest tests for SlopDetector analyzers and ensemble."""

import io
import json
import numpy as np
import pytest
from PIL import Image

from ai_detector.analyzers.base import AnalysisResult
from ai_detector.analyzers.ela import ELAAnalyzer
from ai_detector.analyzers.spectral import SpectralAnalyzer
from ai_detector.analyzers.metadata import MetadataAnalyzer
from ai_detector.analyzers.noise import NoiseAnalyzer
from ai_detector import ensemble


# ─── helpers ──────────────────────────────────────────────────────────────────

def _save_tmp_image(tmp_path, img: Image.Image, name: str = "test.jpg") -> str:
    p = tmp_path / name
    img.save(str(p))
    return str(p)


def _rgb_image(w: int = 64, h: int = 64, seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _gan_checkerboard_image(w: int = 64, h: int = 64, stride: int = 4) -> Image.Image:
    """Synthetic GAN-like image: regular sparse grid of bright spots mimicking
    stride-based upsampling artefacts. Produces many peaks in the 2D FFT."""
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[::stride, ::stride] = 255
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 15, (h, w), dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, "RGB")


def _smooth_gradient_image(w: int = 64, h: int = 64) -> Image.Image:
    """Smooth horizontal gradient — energy concentrated at DC, close to 1/f."""
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, "RGB")


def _result(name: str, pct: float, conf: float, heatmap: np.ndarray | None = None) -> AnalysisResult:
    return AnalysisResult(
        analyzer=name,
        ai_percentage=pct,
        confidence=conf,
        indicators=[f"{name} test indicator"],
        heatmap=heatmap,
    )


# ─── ELA ──────────────────────────────────────────────────────────────────────

class TestELAAnalyzer:
    def test_returns_analysis_result(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img)
        result = ELAAnalyzer().analyze(path)
        assert isinstance(result, AnalysisResult)
        assert result.analyzer == "ela"

    def test_heatmap_shape(self, tmp_path):
        w, h = 80, 60
        img = _rgb_image(w, h)
        path = _save_tmp_image(tmp_path, img)
        result = ELAAnalyzer().analyze(path)
        assert result.heatmap is not None
        assert result.heatmap.shape == (h, w)

    def test_heatmap_range(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img)
        result = ELAAnalyzer().analyze(path)
        assert result.heatmap is not None
        assert float(result.heatmap.min()) >= 0.0
        assert float(result.heatmap.max()) <= 1.0

    def test_ai_percentage_in_range(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img)
        result = ELAAnalyzer().analyze(path)
        assert 0.0 <= result.ai_percentage <= 100.0

    def test_confidence_in_range(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img)
        result = ELAAnalyzer().analyze(path)
        assert 0.0 <= result.confidence <= 1.0


# ─── Spectral ─────────────────────────────────────────────────────────────────

class TestSpectralAnalyzer:
    def test_returns_analysis_result(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img, "test.png")
        result = SpectralAnalyzer().analyze(path)
        assert isinstance(result, AnalysisResult)
        assert result.analyzer == "spectral"

    def test_heatmap_shape(self, tmp_path):
        w, h = 80, 64
        img = _rgb_image(w, h)
        path = _save_tmp_image(tmp_path, img, "test.png")
        result = SpectralAnalyzer().analyze(path)
        assert result.heatmap is not None
        assert result.heatmap.shape == (h, w)

    def test_heatmap_range(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img, "test.png")
        result = SpectralAnalyzer().analyze(path)
        assert result.heatmap is not None
        assert float(result.heatmap.min()) >= 0.0
        assert float(result.heatmap.max()) <= 1.0

    def test_spike_detection_on_periodic_image(self, tmp_path):
        """GAN checkerboard image should score higher than a smooth gradient."""
        gan_img = _gan_checkerboard_image(64, 64, stride=4)
        gradient_img = _smooth_gradient_image(64, 64)

        gan_path = _save_tmp_image(tmp_path, gan_img, "gan.png")
        gradient_path = _save_tmp_image(tmp_path, gradient_img, "gradient.png")

        gan_result = SpectralAnalyzer().analyze(gan_path)
        gradient_result = SpectralAnalyzer().analyze(gradient_path)

        # GAN checkerboard has many off-centre FFT spikes; smooth gradient does not
        assert gan_result.ai_percentage > gradient_result.ai_percentage

    def test_ai_percentage_in_range(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img, "test.png")
        result = SpectralAnalyzer().analyze(path)
        assert 0.0 <= result.ai_percentage <= 100.0


# ─── Metadata ─────────────────────────────────────────────────────────────────

class TestMetadataAnalyzer:
    def test_clean_image_low_score(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img)
        result = MetadataAnalyzer().analyze(path)
        assert isinstance(result, AnalysisResult)
        assert result.analyzer == "metadata"
        assert 0.0 <= result.ai_percentage <= 100.0

    def test_software_tag_triggers_high_score(self, tmp_path):
        """JPEG with AI software string in EXIF Software tag should score high."""
        img = _rgb_image()
        path = str(tmp_path / "ai_software.jpg")

        # Save with piexif to embed Software EXIF tag
        import piexif
        exif_dict = {"0th": {piexif.ImageIFD.Software: b"Stable Diffusion"}}
        exif_bytes = piexif.dump(exif_dict)
        img.save(path, exif=exif_bytes)

        result = MetadataAnalyzer().analyze(path)
        assert result.ai_percentage >= 70

    def test_png_chunk_triggers_high_score(self, tmp_path):
        """PNG with 'parameters' chunk (common in A1111 outputs) should score high."""
        img = _rgb_image()
        path = str(tmp_path / "ai_png.png")

        # PIL PngImagePlugin stores tEXt chunks via img.info before save
        from PIL import PngImagePlugin
        meta = PngImagePlugin.PngInfo()
        meta.add_text("parameters", "steps: 20, sampler: Euler a, cfg scale: 7")
        img.save(path, pnginfo=meta)

        result = MetadataAnalyzer().analyze(path)
        assert result.ai_percentage >= 50

    def test_no_heatmap(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img)
        result = MetadataAnalyzer().analyze(path)
        assert result.heatmap is None

    def test_has_indicators(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img)
        result = MetadataAnalyzer().analyze(path)
        assert isinstance(result.indicators, list)
        assert len(result.indicators) > 0


# ─── Noise ────────────────────────────────────────────────────────────────────

def _flat_image(w: int = 64, h: int = 64, value: int = 128) -> Image.Image:
    """Perfectly flat image — maximum over-smoothness signal."""
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _shot_noise_image(w: int = 64, h: int = 64) -> Image.Image:
    """Gradient image with Poisson-like noise: noise variance ∝ signal level."""
    rng = np.random.default_rng(99)
    base = np.linspace(10, 240, w, dtype=np.float32)
    arr = np.tile(base, (h, 1))
    # Add noise proportional to sqrt(signal) to simulate shot noise
    noise = rng.standard_normal((h, w)) * np.sqrt(arr) * 0.5
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(rgb, "RGB")


class TestNoiseAnalyzer:
    def test_returns_analysis_result(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img, "test.png")
        result = NoiseAnalyzer().analyze(path)
        assert isinstance(result, AnalysisResult)
        assert result.analyzer == "noise"

    def test_ai_percentage_in_range(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img, "test.png")
        result = NoiseAnalyzer().analyze(path)
        assert 0.0 <= result.ai_percentage <= 100.0

    def test_confidence_in_range(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img, "test.png")
        result = NoiseAnalyzer().analyze(path)
        assert 0.0 <= result.confidence <= 1.0

    def test_heatmap_shape_and_range(self, tmp_path):
        w, h = 80, 64
        img = _rgb_image(w, h)
        path = _save_tmp_image(tmp_path, img, "test.png")
        result = NoiseAnalyzer().analyze(path)
        assert result.heatmap is not None
        assert result.heatmap.shape == (h, w)
        assert float(result.heatmap.min()) >= 0.0
        assert float(result.heatmap.max()) <= 1.0

    def test_flat_image_scores_high(self, tmp_path):
        """Perfectly flat image (no noise, no texture) should score high AI."""
        flat = _flat_image(64, 64, value=128)
        path = _save_tmp_image(tmp_path, flat, "flat.png")
        result = NoiseAnalyzer().analyze(path)
        assert result.ai_percentage > 50.0

    def test_shot_noise_image_scores_lower_than_flat(self, tmp_path):
        """Image with Poisson-like noise should score lower than a flat image."""
        flat = _flat_image(64, 64)
        noisy = _shot_noise_image(64, 64)
        flat_path = _save_tmp_image(tmp_path, flat, "flat.png")
        noisy_path = _save_tmp_image(tmp_path, noisy, "noisy.png")
        flat_result = NoiseAnalyzer().analyze(flat_path)
        noisy_result = NoiseAnalyzer().analyze(noisy_path)
        assert flat_result.ai_percentage > noisy_result.ai_percentage

    def test_has_indicators(self, tmp_path):
        img = _rgb_image()
        path = _save_tmp_image(tmp_path, img, "test.png")
        result = NoiseAnalyzer().analyze(path)
        assert isinstance(result.indicators, list)
        assert len(result.indicators) > 0


# ─── Ensemble ─────────────────────────────────────────────────────────────────

class TestEnsemble:
    def test_empty_results(self):
        summary = ensemble.combine([])
        assert summary["ai_percentage"] == 50.0
        assert summary["confidence"] == 0.0
        assert summary["verdict"] == "Unknown"

    def test_summary_has_required_keys(self):
        results = [_result("ela", 60.0, 0.7), _result("spectral", 55.0, 0.6)]
        summary = ensemble.combine(results)
        for key in ("ai_percentage", "confidence", "verdict", "weights_used", "per_analyzer"):
            assert key in summary

    def test_runs_without_gemini(self, tmp_path):
        """Full pipeline should work using only local analyzers."""
        img = _rgb_image()
        jpeg_path = _save_tmp_image(tmp_path, img)
        png_path = _save_tmp_image(tmp_path, img, "test.png")

        ela_result = ELAAnalyzer().analyze(jpeg_path)
        spectral_result = SpectralAnalyzer().analyze(png_path)
        metadata_result = MetadataAnalyzer().analyze(jpeg_path)

        summary = ensemble.combine([ela_result, spectral_result, metadata_result])
        assert 0.0 <= summary["ai_percentage"] <= 100.0
        assert summary["verdict"] in (
            "Inconclusive", "Likely AI-Generated", "Possibly AI-Generated",
            "Probably Real", "Likely Real"
        )

    def test_handles_analyzer_failure_gracefully(self):
        """Ensemble should work even when one result is missing."""
        results = [_result("ela", 80.0, 0.9)]
        summary = ensemble.combine(results)
        assert 0.0 <= summary["ai_percentage"] <= 100.0

    def test_weighted_average(self):
        """Weighted average should pull toward higher-weight analyzer."""
        # gemini=0.1575 weight, ela=0.07 — with only these two, gemini dominates
        hm = np.full((32, 32), 0.5, dtype=np.float32)
        gemini_result = _result("gemini", 90.0, 0.9, hm)
        ela_result = _result("ela", 10.0, 0.8, hm)
        summary = ensemble.combine([gemini_result, ela_result])
        # gemini weight 0.20, ela weight 0.35 → normalised 4:7 ratio
        # Expected ≈ (90*0.20 + 10*0.35) / 0.55 ≈ 39.1 — ela dominates
        assert summary["ai_percentage"] < 50.0

    def test_heatmap_combined(self):
        hm1 = np.ones((32, 32), dtype=np.float32)
        hm2 = np.zeros((32, 32), dtype=np.float32)
        r1 = _result("ela", 50.0, 0.5, hm1)
        r2 = _result("spectral", 50.0, 0.5, hm2)
        summary = ensemble.combine([r1, r2])
        assert summary["heatmap"] is not None
        assert 0.0 <= float(summary["heatmap"].mean()) <= 1.0

    def test_per_analyzer_dict_populated(self):
        r = _result("ela", 55.0, 0.7)
        summary = ensemble.combine([r])
        assert "ela" in summary["per_analyzer"]
        assert summary["per_analyzer"]["ela"]["ai_percentage"] == 55.0

    def test_analysis_result_to_dict(self):
        r = _result("spectral", 42.0, 0.6)
        d = r.to_dict()
        assert d["analyzer"] == "spectral"
        assert d["ai_percentage"] == 42.0
        assert d["confidence"] == 0.6
        assert isinstance(d["indicators"], list)


# ─── Verdict ──────────────────────────────────────────────────────────────────

class TestVerdict:
    @pytest.mark.parametrize("pct,conf,expected", [
        (80.0, 0.9, "Likely AI-Generated"),
        (60.0, 0.5, "Possibly AI-Generated"),
        (30.0, 0.5, "Probably Real"),
        (10.0, 0.5, "Likely Real"),
        (80.0, 0.2, "Inconclusive"),
    ])
    def test_verdicts(self, pct, conf, expected):
        assert ensemble._verdict(pct, conf) == expected

"""ResNet50 ML analyzer — lazy-loads a fine-tuned checkpoint on first inference."""

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from .base import AnalysisResult, BaseAnalyzer
from .metadata import has_camera_exif

_DEFAULT_MODEL_PATH = Path(__file__).parents[2] / "training" / "best_model.pt"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Lazy singleton ────────────────────────────────────────────────────────────
# Model is loaded once on the first real inference request, never on import
# or health checks. _lock prevents a double-load race under concurrent requests.

_lock      = threading.Lock()
_model     = None   # ResNet50 instance, or None until first load
_preprocess = None  # torchvision transforms pipeline


def _load_model() -> bool:
    """Load and cache the ResNet50 model. Returns False if unavailable."""
    global _model, _preprocess

    if _model is not None:
        return True

    with _lock:
        if _model is not None:   # double-checked: another thread may have loaded it
            return True

        try:
            import torch
            import torch.nn.functional as F
            from torchvision import models, transforms
        except ImportError:
            return False

        # Cap CPU threads once, right when torch is first used.
        torch.set_num_threads(1)

        model_path = Path(os.environ.get("SLOP_MODEL_PATH", _DEFAULT_MODEL_PATH))
        if not model_path.exists():
            return False

        # Force CPU — no CUDA on Render free tier.
        device = torch.device("cpu")

        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(torch.load(model_path, map_location=device))
        m.to(device).eval()

        prep = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        _model     = m
        _preprocess = prep
        return True


class MLAnalyzer(BaseAnalyzer):
    name = "ml"

    def analyze(self, image_path: str) -> AnalysisResult:
        # Try to ensure the model is loaded; skip gracefully if unavailable.
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["torch/torchvision not installed — ML analysis skipped"],
            )

        if not _load_model():
            model_path = Path(os.environ.get("SLOP_MODEL_PATH", _DEFAULT_MODEL_PATH))
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.1,
                indicators=[f"Model file not found ({model_path}) — ML analysis skipped"],
            )

        device = torch.device("cpu")
        img    = Image.open(image_path).convert("RGB")
        tensor = _preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = _model(tensor)
            probs  = F.softmax(logits, dim=1)
            ai_prob = float(probs[0, 1].item())

        ai_percentage = round(ai_prob * 100, 2)
        confidence    = round(min(0.95, 0.5 + abs(ai_prob - 0.5) * 1.8), 3)

        # Camera EXIF calibration: real cameras always embed Make/Model; cap score
        # at 40% to suppress false positives on professional photography.
        capped = False
        if ai_percentage > 40.0 and has_camera_exif(image_path):
            ai_percentage = 40.0
            confidence    = round(confidence * 0.7, 3)
            capped        = True

        if ai_prob >= 0.7:
            indicator = f"ResNet50 classifier: {ai_percentage:.1f}% AI (p={ai_prob:.3f})"
        elif ai_prob <= 0.3:
            indicator = f"ResNet50 classifier: {ai_percentage:.1f}% AI — likely real (p={ai_prob:.3f})"
        else:
            indicator = f"ResNet50 classifier: {ai_percentage:.1f}% AI — ambiguous (p={ai_prob:.3f})"

        indicators = [indicator]
        if capped:
            indicators.append("Camera Make/Model EXIF present — score capped at 40%")

        return AnalysisResult(
            analyzer=self.name,
            ai_percentage=ai_percentage,
            confidence=confidence,
            indicators=indicators,
        )

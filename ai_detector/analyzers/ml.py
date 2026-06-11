"""ResNet50 ML analyzer — loads a fine-tuned checkpoint and runs inference."""

import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .base import AnalysisResult, BaseAnalyzer
from .metadata import has_camera_exif

log = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).parents[2] / "training" / "best_model.pt"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class MLAnalyzer(BaseAnalyzer):
    name = "ml"

    def analyze(self, image_path: str) -> AnalysisResult:
        try:
            import torch
            import torch.nn.functional as F
            from torchvision import models, transforms
        except ImportError:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["torch/torchvision not installed — ML analysis skipped"],
            )

        model_path = Path(os.environ.get("SLOP_MODEL_PATH", _DEFAULT_MODEL_PATH))
        log.info("MLAnalyzer: looking for model at %s (exists=%s)", model_path, model_path.exists())
        if not model_path.exists():
            log.warning("MLAnalyzer: model file not found at %s — skipping ML analysis", model_path)
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.1,
                indicators=[f"Model file not found ({model_path}) — ML analysis skipped"],
            )

        # A ~130-byte file starting with "version https://git-lfs..." means the
        # checkout has the Git LFS pointer instead of the real 94MB checkpoint.
        if model_path.stat().st_size < 1024:
            head = model_path.read_bytes()[:40]
            if head.startswith(b"version https://git-lfs"):
                log.error(
                    "MLAnalyzer: %s is a Git LFS pointer (%d bytes), not the model — "
                    "run 'git lfs pull' or ensure LFS objects are pushed to this remote",
                    model_path, model_path.stat().st_size,
                )
                return AnalysisResult(
                    analyzer=self.name,
                    ai_percentage=50.0,
                    confidence=0.0,
                    indicators=[f"Model file is a Git LFS pointer ({model_path}) — ML analysis skipped"],
                )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("MLAnalyzer: loading model from %s on device=%s", model_path, device)
        try:
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device).eval()
        except Exception:
            log.exception("MLAnalyzer: failed to load model from %s", model_path)
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=[f"Model load failed ({model_path}) — ML analysis skipped"],
            )
        log.info("MLAnalyzer: model loaded successfully")

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

        img = Image.open(image_path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            ai_prob = float(probs[0, 1].item())

        ai_percentage = round(ai_prob * 100, 2)
        confidence = round(min(0.95, 0.5 + abs(ai_prob - 0.5) * 1.8), 3)

        # Camera EXIF calibration: real cameras always embed Make/Model; cap score
        # at 40% to suppress false positives on professional photography.
        capped = False
        if ai_percentage > 40.0 and has_camera_exif(image_path):
            ai_percentage = 40.0
            confidence = round(confidence * 0.7, 3)
            capped = True

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

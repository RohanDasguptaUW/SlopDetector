"""ResNet50 ML analyzer — loads a fine-tuned checkpoint and runs inference."""

import logging
import os
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .base import AnalysisResult, BaseAnalyzer
from .metadata import has_camera_exif

log = logging.getLogger(__name__)

_DEFAULT_MODEL_PATH = Path(__file__).parents[2] / "training" / "best_model.pt"
_GDRIVE_FILE_ID = "1pJTM4dlQwJDvKLA74yxsPlOCS-B2Njr3"
_GDRIVE_BASE_URL = "https://drive.google.com/uc"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _download_model(dest: Path) -> None:
    """Fetch best_model_v3.pt from Google Drive and save to *dest*.

    Google Drive serves a virus-scan HTML confirmation page for files over ~25 MB.
    We detect that by content-type, extract the confirm token, and re-request the
    actual binary.  The file is streamed in 1 MB chunks to a temp path, then
    atomically renamed so a partial download never leaves a corrupt checkpoint.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    params = {"export": "download", "id": _GDRIVE_FILE_ID}
    session = requests.Session()

    log.info(
        "MLAnalyzer: model not found — downloading from Google Drive "
        "(file_id=%s) → %s", _GDRIVE_FILE_ID, dest
    )

    # First request (streaming) — large files get an HTML confirmation page.
    resp = session.get(_GDRIVE_BASE_URL, params=params, stream=True, timeout=30)
    resp.raise_for_status()

    if "text/html" in resp.headers.get("content-type", ""):
        # Confirmation page is small HTML; consume it to extract the token.
        body = resp.content.decode("utf-8", errors="replace")
        m = re.search(r'confirm=([0-9A-Za-z_\-]+)', body)
        token = m.group(1) if m else "t"
        log.info("MLAnalyzer: Google Drive virus-scan confirmation (token=%r) — retrying", token)
        params["confirm"] = token
        resp = session.get(_GDRIVE_BASE_URL, params=params, stream=True, timeout=30)
        resp.raise_for_status()

    # Stream to a sibling temp file, then rename atomically.
    tmp = dest.with_suffix(".downloading")
    try:
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
                f.write(chunk)
                downloaded += len(chunk)
        log.info(
            "MLAnalyzer: download complete — %.1f MB written, moving to %s",
            downloaded / 1e6, dest,
        )
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


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

        # Download if missing or if we only have the Git LFS pointer stub.
        needs_download = not model_path.exists()
        if not needs_download and model_path.stat().st_size < 1024:
            head = model_path.read_bytes()[:40]
            if head.startswith(b"version https://git-lfs"):
                log.warning(
                    "MLAnalyzer: %s is a Git LFS pointer (%d bytes) — will download real checkpoint",
                    model_path, model_path.stat().st_size,
                )
                needs_download = True

        if needs_download:
            try:
                _download_model(model_path)
            except Exception:
                log.exception("MLAnalyzer: failed to download model from Google Drive")
                return AnalysisResult(
                    analyzer=self.name,
                    ai_percentage=50.0,
                    confidence=0.0,
                    indicators=["Model download failed — ML analysis skipped"],
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

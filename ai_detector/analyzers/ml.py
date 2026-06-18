"""ResNet50 ML analyzer — loads a fine-tuned checkpoint and runs inference."""

import html
import logging
import os
import re
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


def _looks_like_html(first_bytes: bytes) -> bool:
    head = first_bytes[:64].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _resolve_confirm_request(session, body: str):
    """Given the HTML virus-scan page, return (url, params) for the real download.

    Modern Google Drive serves a <form id="download-form"> that POSTs/GETs to
    drive.usercontent.google.com with hidden inputs (id, export, confirm, uuid).
    Older variants embed a `confirm=<token>` link back on the uc endpoint.
    We try the form first, then fall back to the legacy token.
    """
    form = re.search(
        r'<form[^>]*id="download-form"[^>]*action="([^"]+)"', body, re.IGNORECASE
    )
    if form:
        action = html.unescape(form.group(1))
        inputs = re.findall(
            r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', body, re.IGNORECASE
        )
        params = {name: html.unescape(val) for name, val in inputs}
        params.setdefault("id", _GDRIVE_FILE_ID)
        params.setdefault("export", "download")
        params.setdefault("confirm", "t")
        return action, params

    # Legacy fallback: a confirm token in a link or the download_warning cookie.
    m = re.search(r'confirm=([0-9A-Za-z_\-]+)', body)
    token = m.group(1) if m else "t"
    for k, v in session.cookies.items():
        if k.startswith("download_warning"):
            token = v
    return _GDRIVE_BASE_URL, {"export": "download", "id": _GDRIVE_FILE_ID, "confirm": token}


def _download_model(dest: Path) -> None:
    """Fetch best_model_v3.pt from Google Drive and save to *dest*.

    Google Drive serves an HTML virus-scan confirmation page for files over
    ~25 MB instead of the binary. We follow the confirmation form to the real
    download URL, stream the binary in 1 MB chunks to a temp path, verify it is
    not itself an HTML page, then atomically rename so a partial or bogus
    download never leaves a corrupt checkpoint in place.
    """
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    log.info(
        "MLAnalyzer: model not found — downloading from Google Drive "
        "(file_id=%s) → %s", _GDRIVE_FILE_ID, dest
    )

    url = _GDRIVE_BASE_URL
    params = {"export": "download", "id": _GDRIVE_FILE_ID}
    resp = session.get(url, params=params, stream=True, timeout=60)
    resp.raise_for_status()

    if "text/html" in resp.headers.get("content-type", "").lower():
        body = resp.content.decode("utf-8", errors="replace")
        url, params = _resolve_confirm_request(session, body)
        log.info("MLAnalyzer: Google Drive virus-scan page — following confirm to %s", url)
        resp = session.get(url, params=params, stream=True, timeout=60)
        resp.raise_for_status()

    ctype = resp.headers.get("content-type", "").lower()
    if "text/html" in ctype:
        raise RuntimeError(
            f"Google Drive returned HTML, not the model binary (content-type={ctype!r}). "
            "The file may be private or the confirm flow changed."
        )

    # Stream to a sibling temp file, then rename atomically.
    tmp = dest.with_suffix(".downloading")
    try:
        downloaded = 0
        first = b""
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
                if not chunk:
                    continue
                if not first:
                    first = chunk
                    if _looks_like_html(first):
                        raise RuntimeError(
                            "Downloaded content is an HTML page, not the model binary "
                            "(Google Drive confirm flow failed)."
                        )
                f.write(chunk)
                downloaded += len(chunk)
        if downloaded < 1 << 20:  # sanity: real checkpoint is ~94 MB
            raise RuntimeError(f"Downloaded file is implausibly small ({downloaded} bytes).")
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

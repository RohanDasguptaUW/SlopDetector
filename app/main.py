"""SlopDetector — FastAPI backend.

Deployment modes
----------------
Full mode  (default / HF Space):
    No HF_SPACE_URL set. All 5 analyzers run locally. ResNet50 is lazy-loaded
    on first inference — torch is never imported at startup.

Split mode (Render free tier):
    Both HF_SPACE_URL and ML_API_SECRET env vars are set.
    Only the 4 lightweight analyzers run locally; the ResNet50 score is
    fetched from the HF Space instance via /ml-score. Torch is never imported.
"""

import base64
import gc
import io
import logging
import os
import sys
import tempfile
from pathlib import Path

import httpx
from PIL import Image as _Pil
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parents[1]))

from ai_detector import ensemble
from ai_detector import heatmap as heatmap_mod
from ai_detector.analyzers.base import AnalysisResult
from ai_detector.analyzers.ela import ELAAnalyzer
from ai_detector.analyzers.spectral import SpectralAnalyzer
from ai_detector.analyzers.metadata import MetadataAnalyzer
from ai_detector.analyzers.noise import NoiseAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("slopdetector")

_HF_SPACE_URL  = os.environ.get("HF_SPACE_URL", "").rstrip("/")
_ML_API_SECRET = os.environ.get("ML_API_SECRET", "")

# Split mode: Render POSTs to HF Space for ResNet50; torch is never touched here.
# Full mode:  MLAnalyzer is present; torch lazy-loads on first real request.
_SPLIT_MODE = bool(_HF_SPACE_URL and _ML_API_SECRET)

if _SPLIT_MODE:
    log.info("Split mode active — ML scores will be fetched from %s", _HF_SPACE_URL)
    _ANALYZERS = [ELAAnalyzer(), SpectralAnalyzer(), MetadataAnalyzer(), NoiseAnalyzer()]
else:
    from ai_detector.analyzers.ml import MLAnalyzer  # torch not imported until first call
    _ANALYZERS = [ELAAnalyzer(), SpectralAnalyzer(), MetadataAnalyzer(), NoiseAnalyzer(), MLAnalyzer()]

_STATIC = Path(__file__).parent / "static"
_MAX_SIDE = 1024


def _shrink(raw: bytes, suffix: str) -> tuple[bytes, str]:
    """Down-scale to _MAX_SIDE on the longest side, preserving aspect ratio.
    Returns original bytes unchanged when the image is already within the limit.
    When resizing is needed the output is always JPEG (quality 95); returns ".jpg"."""
    with _Pil.open(io.BytesIO(raw)) as img:
        w, h = img.size
        if max(w, h) <= _MAX_SIDE:
            return raw, suffix
        scale = _MAX_SIDE / max(w, h)
        img = img.convert("RGB")
        img = img.resize((round(w * scale), round(h * scale)), _Pil.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), ".jpg"


app = FastAPI(title="SlopDetector", docs_url=None, redoc_url=None)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Lightweight liveness probe — never triggers model load."""
    return {"status": "ok"}


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(_STATIC / "index.html", media_type="text/html")


# ── Internal ML-only endpoint (HF Space) ─────────────────────────────────────

@app.post("/ml-score")
async def ml_score(
    image: UploadFile = File(...),
    x_api_secret: str | None = Header(default=None),
):
    """Run only the ResNet50 analyzer and return its score.

    This endpoint is for internal service-to-service calls only.
    Requires a matching X-API-Secret header.
    Only functional on the HF Space instance (full mode).
    """
    if not _ML_API_SECRET:
        raise HTTPException(status_code=503, detail="ML scoring is not configured on this instance.")
    if x_api_secret != _ML_API_SECRET:
        raise HTTPException(status_code=403, detail="Forbidden.")
    if _SPLIT_MODE:
        # This instance is the lightweight Render app; it doesn't run the model.
        raise HTTPException(status_code=503, detail="ML inference is not available on this instance.")

    suffix = Path(image.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        img_path = tmp.name

    try:
        from ai_detector.analyzers.ml import MLAnalyzer
        result = MLAnalyzer().analyze(img_path)
        return {"ml_score": result.ai_percentage, "confidence": result.confidence}
    finally:
        try:
            os.unlink(img_path)
        except OSError:
            pass
        gc.collect()


# ── Split-mode helper ─────────────────────────────────────────────────────────

async def _fetch_ml_score(image_bytes: bytes, filename: str) -> AnalysisResult | None:
    """POST image to HF Space /ml-score (5 s timeout).

    Returns None on any failure; the ensemble will reweight automatically
    when ML is absent, so callers can treat None as a graceful skip.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{_HF_SPACE_URL}/ml-score",
                files={"image": (filename, image_bytes, "application/octet-stream")},
                headers={"X-API-Secret": _ML_API_SECRET},
            )
            resp.raise_for_status()
            data = resp.json()
        return AnalysisResult(
            analyzer="ml",
            ai_percentage=float(data["ml_score"]),
            confidence=float(data["confidence"]),
            indicators=[f"ResNet50 via HF Space: {data['ml_score']:.1f}% AI"],
        )
    except Exception as exc:
        log.warning("HF Space /ml-score call failed (%s) — continuing without ML score", exc)
        return None


# ── Main analyse endpoint ─────────────────────────────────────────────────────

@app.post("/analyse")
async def analyse(image: UploadFile = File(...)):
    image_bytes = await image.read()
    filename    = image.filename or "image.jpg"
    suffix      = Path(filename).suffix or ".jpg"

    # Noise analysis needs the original full-resolution pixels — resizing destroys
    # the shot-noise pattern that distinguishes camera sensor noise from AI output.
    # All other analyzers get the down-scaled copy to keep memory usage low.
    shrunk_bytes, shrunk_suffix = _shrink(image_bytes, suffix)

    with tempfile.NamedTemporaryFile(suffix=shrunk_suffix, delete=False) as tmp:
        tmp.write(shrunk_bytes)
        img_path = tmp.name

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_orig:
        tmp_orig.write(image_bytes)
        orig_path = tmp_orig.name

    try:
        results: list[AnalysisResult] = []
        for analyzer in _ANALYZERS:
            try:
                path = orig_path if analyzer.name == "noise" else img_path
                results.append(analyzer.analyze(path))
            except Exception:
                pass

        if _SPLIT_MODE:
            ml_result = await _fetch_ml_score(image_bytes, filename)
            if ml_result is not None:
                results.append(ml_result)
            # On fallback (ml_result is None), ensemble.combine() normalises the
            # remaining four analyzers' weights to sum to 1 automatically.

        if not results:
            raise HTTPException(status_code=422, detail="No analyzers could process this image.")

        summary = ensemble.combine(results)
        weights  = summary["weights_used"]

        heatmap_b64 = None
        if summary.get("heatmap") is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as hm:
                hm_path = hm.name
            try:
                heatmap_mod.generate(img_path, summary["heatmap"], summary["ai_percentage"], hm_path)
                with open(hm_path, "rb") as f:
                    heatmap_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()
            finally:
                try:
                    os.unlink(hm_path)
                except OSError:
                    pass

        return {
            "ai_percentage": round(summary["ai_percentage"], 1),
            "confidence":    round(summary["confidence"], 3),
            "verdict":       summary["verdict"],
            "heatmap_b64":   heatmap_b64,
            "results": [
                {
                    "analyzer":      r.analyzer,
                    "ai_percentage": round(r.ai_percentage, 1),
                    "confidence":    round(r.confidence, 3),
                    "weight":        round(weights.get(r.analyzer, 0), 3),
                    "indicators":    r.indicators,
                }
                for r in results
            ],
        }
    finally:
        for p in (img_path, orig_path):
            try:
                os.unlink(p)
            except OSError:
                pass
        gc.collect()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

"""SlopDetector — FastAPI backend."""

import base64
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

sys.path.insert(0, str(Path(__file__).parents[1]))

from ai_detector import ensemble
from ai_detector import heatmap as heatmap_mod
from ai_detector.analyzers.ela import ELAAnalyzer
from ai_detector.analyzers.spectral import SpectralAnalyzer
from ai_detector.analyzers.metadata import MetadataAnalyzer
from ai_detector.analyzers.noise import NoiseAnalyzer
from ai_detector.analyzers.ml import MLAnalyzer

_ANALYZERS = [ELAAnalyzer(), SpectralAnalyzer(), MetadataAnalyzer(), NoiseAnalyzer(), MLAnalyzer()]

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="SlopDetector")


@app.get("/")
async def index():
    return FileResponse(_STATIC / "index.html", media_type="text/html")


@app.post("/analyse")
async def analyse(image: UploadFile = File(...)):
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await image.read())
        image_path = tmp.name

    try:
        results = []
        for analyzer in _ANALYZERS:
            try:
                results.append(analyzer.analyze(image_path))
            except Exception:
                pass

        if not results:
            raise HTTPException(status_code=422, detail="No analyzers could process this image.")

        summary = ensemble.combine(results)
        weights = summary["weights_used"]

        heatmap_b64 = None
        if summary.get("heatmap") is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as hm:
                hm_path = hm.name
            try:
                heatmap_mod.generate(image_path, summary["heatmap"], summary["ai_percentage"], hm_path)
                with open(hm_path, "rb") as f:
                    heatmap_b64 = "data:image/png;base64," + base64.b64encode(f.read()).decode()
            finally:
                try:
                    os.unlink(hm_path)
                except OSError:
                    pass

        return {
            "ai_percentage": round(summary["ai_percentage"], 1),
            "confidence": round(summary["confidence"], 3),
            "verdict": summary["verdict"],
            "heatmap_b64": heatmap_b64,
            "results": [
                {
                    "analyzer": r.analyzer,
                    "ai_percentage": round(r.ai_percentage, 1),
                    "confidence": round(r.confidence, 3),
                    "weight": round(weights.get(r.analyzer, 0), 3),
                    "indicators": r.indicators,
                }
                for r in results
            ],
        }
    finally:
        try:
            os.unlink(image_path)
        except OSError:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

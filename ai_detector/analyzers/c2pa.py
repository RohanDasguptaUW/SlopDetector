"""C2PA Content Credentials analyzer."""

import json
import os
from typing import Any

from .base import AnalysisResult, BaseAnalyzer

_AI_SOURCE_TYPES = {
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmicMedia",
}

_CAMERA_SOURCE_TYPES = {
    "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
    "http://cv.iptc.org/newscodes/digitalsourcetype/negativeFilm",
    "http://cv.iptc.org/newscodes/digitalsourcetype/positiveFilm",
}


def _find_digital_source_types(manifest_store: dict[str, Any]) -> list[str]:
    """Walk all manifests and collect every digitalSourceType value found."""
    found: list[str] = []
    for manifest in manifest_store.get("manifests", {}).values():
        for assertion in manifest.get("assertions", []):
            data = assertion.get("data", {})
            # c2pa.actions assertion
            for action in data.get("actions", []):
                dst = action.get("digitalSourceType") or action.get("digital_source_type")
                if dst:
                    found.append(dst)
            # stds.schema-org.CreativeWork or similar
            dst = data.get("digitalSourceType") or data.get("digital_source_type")
            if dst:
                found.append(dst)
    return found


class C2PAAnalyzer(BaseAnalyzer):
    name = "c2pa"

    def analyze(self, image_path: str) -> AnalysisResult:
        try:
            import c2pa  # type: ignore[import]
        except ImportError:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["c2pa-python not installed — C2PA analysis skipped"],
            )

        try:
            reader = c2pa.Reader.from_file(image_path)
            store_json = reader.json()
        except Exception:
            # No C2PA manifest present
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.2,
                indicators=["No C2PA provenance data found"],
            )

        try:
            store: dict[str, Any] = json.loads(store_json) if isinstance(store_json, str) else store_json
        except (json.JSONDecodeError, TypeError):
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.1,
                indicators=["C2PA manifest present but could not be parsed"],
            )

        source_types = _find_digital_source_types(store)

        is_ai = any(t in _AI_SOURCE_TYPES for t in source_types)
        is_camera = any(t in _CAMERA_SOURCE_TYPES for t in source_types)

        if is_ai:
            matched = [t for t in source_types if t in _AI_SOURCE_TYPES]
            short = [t.rsplit("/", 1)[-1] for t in matched]
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=100.0,
                confidence=0.95,
                indicators=[f"C2PA manifest declares AI generation ({', '.join(short)})"],
            )

        if is_camera:
            matched = [t for t in source_types if t in _CAMERA_SOURCE_TYPES]
            short = [t.rsplit("/", 1)[-1] for t in matched]
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=0.0,
                confidence=0.95,
                indicators=[f"C2PA manifest declares camera capture ({', '.join(short)})"],
            )

        # Manifest present but no recognised source type
        return AnalysisResult(
            analyzer=self.name,
            ai_percentage=50.0,
            confidence=0.3,
            indicators=["C2PA manifest found but source type unrecognised — treating as inconclusive"],
        )

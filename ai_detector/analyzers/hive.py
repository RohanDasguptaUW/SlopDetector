"""Hive AI detection analyzer — calls the Hive moderation API."""

import mimetypes
import os

from .base import AnalysisResult, BaseAnalyzer

_HIVE_ENDPOINT = "https://api.thehive.ai/api/v2/task/sync"


def _parse_ai_score(response_json: dict) -> float | None:
    """Extract the ai_generated score (0–1) from the Hive API response."""
    try:
        for status in response_json.get("status", []):
            for output in status.get("response", {}).get("output", []):
                for cls in output.get("classes", []):
                    if cls.get("class") == "ai_generated":
                        return float(cls["score"])
    except (KeyError, TypeError, ValueError):
        pass
    return None


class HiveAnalyzer(BaseAnalyzer):
    name = "hive"

    def analyze(self, image_path: str) -> AnalysisResult:
        try:
            import requests  # type: ignore[import]
        except ImportError:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["requests not installed — Hive analysis skipped"],
            )

        api_key = os.environ.get("HIVE_API_KEY", "")
        if not api_key:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["HIVE_API_KEY not set — Hive analysis skipped"],
            )

        mime, _ = mimetypes.guess_type(image_path)
        if not mime or not mime.startswith("image/"):
            mime = "application/octet-stream"

        try:
            with open(image_path, "rb") as fh:
                response = requests.post(
                    _HIVE_ENDPOINT,
                    headers={"Authorization": f"Token {api_key}"},
                    files={"media": (os.path.basename(image_path), fh, mime)},
                    timeout=30,
                )
        except requests.exceptions.RequestException as exc:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=[f"Hive API request failed: {exc}"],
            )

        if response.status_code != 200:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=[f"Hive API returned HTTP {response.status_code}"],
            )

        try:
            data = response.json()
        except ValueError:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["Hive API returned non-JSON response"],
            )

        score = _parse_ai_score(data)
        if score is None:
            return AnalysisResult(
                analyzer=self.name,
                ai_percentage=50.0,
                confidence=0.0,
                indicators=["Hive API response missing ai_generated class"],
            )

        ai_percentage = round(score * 100, 2)
        # Hive confidence is proportional to how far the score is from 0.5
        confidence = round(min(0.95, 0.5 + abs(score - 0.5) * 1.5), 3)

        if score >= 0.7:
            label = f"Hive classifier: {ai_percentage:.1f}% AI-generated (score={score:.3f})"
        elif score <= 0.3:
            label = f"Hive classifier: {ai_percentage:.1f}% AI-generated — likely real (score={score:.3f})"
        else:
            label = f"Hive classifier: {ai_percentage:.1f}% AI-generated — ambiguous (score={score:.3f})"

        return AnalysisResult(
            analyzer=self.name,
            ai_percentage=ai_percentage,
            confidence=confidence,
            indicators=[label],
        )

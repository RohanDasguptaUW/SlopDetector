"""Base types for analyzers."""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AnalysisResult:
    analyzer: str
    ai_percentage: float  # 0–100
    confidence: float     # 0–1
    indicators: list[str] = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None  # float32 0–1, shape (H, W)

    def to_dict(self) -> dict:
        return {
            "analyzer": self.analyzer,
            "ai_percentage": round(self.ai_percentage, 2),
            "confidence": round(self.confidence, 3),
            "indicators": self.indicators,
            "has_heatmap": self.heatmap is not None,
        }


class BaseAnalyzer:
    name: str = "base"

    def analyze(self, image_path: str) -> AnalysisResult:
        raise NotImplementedError

---
title: SlopDetector
emoji: 🔍
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: Detect AI-generated images using forensic analysis
---

# SlopDetector

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen)

Estimate what percentage of an image is AI-generated using a multi-signal ensemble of forensic analyzers.

---

## Architecture

```
ai-detect <images>
       │
       ▼
 _collect_images()          glob / dir expansion, dedup
       │
       ├──► ELAAnalyzer        Error Level Analysis (JPEG re-save diff)
       ├──► SpectralAnalyzer   2D FFT — GAN checkerboard spike detection
       ├──► MetadataAnalyzer   EXIF / PNG chunk AI-software fingerprints
       └──► ClaudeAnalyzer     Vision LLM — 3×3 spatial grid scores
                │
                ▼
          ensemble.combine()   Weighted average (claude 50%, ela 25%,
                │               spectral 15%, metadata 10%)
                ▼
          verdict + heatmap + HTML report
```

## Analyzers

| Analyzer | Signal | Weight |
|---|---|---|
| **ELA** | Re-saves at JPEG q95 and measures per-pixel diff amplitude. Low coefficient of variation → suspiciously uniform texture. | 25% |
| **Spectral** | 2D FFT radial power spectrum. Off-centre spikes above 3σ baseline indicate GAN stride artefacts; deviation from 1/f law scores higher. | 15% |
| **Metadata** | EXIF `Software` tag and PNG `parameters`/`prompt`/`workflow` chunks. Matches against known AI tool strings. | 10% |
| **Claude** | Sends image to Claude vision API with a structured prompt. Returns an `ai_percentage`, `confidence`, and a 3×3 spatial grid of 0–10 scores that becomes the heatmap. | 50% |

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+. The Claude analyzer requires `ANTHROPIC_API_KEY` to be set.

## Usage

```bash
# Analyse a single image (uses Claude by default)
ai-detect photo.jpg

# Batch analyse a directory, skip Claude, emit JSON
ai-detect images/ --no-claude --json-out

# Generate heatmap PNGs and HTML reports, save to results/
ai-detect *.png --heatmap --report -o results/

# Use a different Claude model
ai-detect suspicious.webp --model claude-opus-4-7
```

### CLI flags

| Flag | Description |
|---|---|
| `--no-claude` | Skip the Claude API call (faster, free, less accurate) |
| `--heatmap` | Save a side-by-side false-colour heatmap PNG |
| `--report` | Save a self-contained dark-theme HTML report with embedded images |
| `-o / --output DIR` | Output directory (default: `slopdetector_output/`) |
| `--json-out` | Print full JSON summary to stdout |
| `--model MODEL` | Claude model string (default: `claude-sonnet-4-6`) |

### Python API

```python
from ai_detector.analyzers.ela import ELAAnalyzer
from ai_detector.analyzers.spectral import SpectralAnalyzer
from ai_detector.analyzers.metadata import MetadataAnalyzer
from ai_detector import ensemble

results = [
    ELAAnalyzer().analyze("photo.jpg"),
    SpectralAnalyzer().analyze("photo.jpg"),
    MetadataAnalyzer().analyze("photo.jpg"),
]
summary = ensemble.combine(results)
print(summary["verdict"], summary["ai_percentage"])
```

## Output

### Console table

```
╭──────────────────────────────────────────────────────╮
│                     photo.jpg                        │
├───────────┬────────┬────────────┬───────────────────┤
│ Analyzer  │  AI %  │ Confidence │ Key Indicators    │
├───────────┼────────┼────────────┼───────────────────┤
│ ela       │  72.3% │       0.68 │ Low ELA amplitude │
│ spectral  │  15.1% │       0.37 │ Spectral profile… │
│ metadata  │  80.0% │       0.90 │ AI software tag   │
│ claude    │  85.0% │       0.92 │ Perfect skin tex… │
├───────────┼────────┼────────────┼───────────────────┤
│ ENSEMBLE  │  79.1% │       0.85 │ Likely AI-Generated│
╰───────────┴────────┴────────────┴───────────────────╯
```

### JSON (`--json-out`)

```json
{
  "photo.jpg": {
    "ai_percentage": 79.1,
    "confidence": 0.85,
    "verdict": "Likely AI-Generated",
    "weights_used": {"claude": 0.5, "ela": 0.25, "spectral": 0.15, "metadata": 0.1},
    "per_analyzer": { ... }
  }
}
```

## Running Tests

```bash
python3 -m pytest tests/ -v
```

## Limitations

- ELA and spectral signals are heuristic — high JPEG compression or aggressive resizing reduces their reliability.
- Metadata signals require the original file; screenshots, re-saves, or social-media re-encoding strip EXIF/PNG chunks.
- Claude's vision analysis reflects the model's training and can produce false positives on painterly or heavily processed real photographs.
- Results are probabilistic. Do not use as sole evidence of AI generation.

## License

MIT

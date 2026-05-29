"""Gradio web UI for SlopDetector."""

import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from ai_detector import ensemble
from ai_detector import heatmap as heatmap_mod
from ai_detector.analyzers.ela import ELAAnalyzer
from ai_detector.analyzers.spectral import SpectralAnalyzer
from ai_detector.analyzers.metadata import MetadataAnalyzer
from ai_detector.analyzers.noise import NoiseAnalyzer
from ai_detector.analyzers.ml import MLAnalyzer

_ANALYZERS = [ELAAnalyzer(), SpectralAnalyzer(), MetadataAnalyzer(), NoiseAnalyzer(), MLAnalyzer()]

# Temp files created for heatmap PNGs; cleaned up between requests
_last_heatmap: list[str] = []


def _verdict_color(ai_pct: float) -> str:
    if ai_pct >= 75:
        return "#e74c3c"
    if ai_pct >= 45:
        return "#e67e22"
    if ai_pct >= 20:
        return "#f1c40f"
    return "#27ae60"


def analyze(image_path: str | None):
    # Clean up previous heatmap temp file
    if _last_heatmap:
        try:
            os.unlink(_last_heatmap.pop())
        except OSError:
            pass

    if image_path is None:
        empty_df = pd.DataFrame(columns=["Analyzer", "AI %", "Confidence", "Weight", "Top indicator"])
        return (
            "<div style='text-align:center;color:#888;padding:40px;'>Upload an image to analyse.</div>",
            empty_df,
            None,
            "",
        )

    results = []
    for analyzer in _ANALYZERS:
        try:
            results.append(analyzer.analyze(image_path))
        except Exception:
            pass

    if not results:
        return "<div style='color:#e74c3c;'>No analyzers could process this image.</div>", pd.DataFrame(), None, ""

    summary = ensemble.combine(results)
    ai_pct = summary["ai_percentage"]
    confidence = summary["confidence"]
    verdict = summary["verdict"]
    weights = summary["weights_used"]

    color = _verdict_color(ai_pct)

    verdict_html = f"""
<div style="background:#16213e;border-radius:12px;padding:28px;text-align:center;font-family:system-ui,sans-serif;">
  <div style="font-size:72px;font-weight:900;color:{color};line-height:1;">{ai_pct:.1f}%</div>
  <div style="margin-top:10px;">
    <span style="background:{color};color:#fff;padding:6px 18px;border-radius:20px;
                 font-weight:700;font-size:17px;">{verdict}</span>
  </div>
  <div style="margin-top:14px;color:#aaa;font-size:14px;">Confidence: {confidence:.0%}</div>
</div>"""

    # Breakdown table
    rows = []
    for r in results:
        w = weights.get(r.analyzer, 0)
        rows.append({
            "Analyzer": r.analyzer,
            "AI %": f"{r.ai_percentage:.1f}",
            "Confidence": f"{r.confidence:.2f}",
            "Weight": f"{w:.0%}",
            "Top indicator": r.indicators[0] if r.indicators else "",
        })
    df = pd.DataFrame(rows)

    # Heatmap
    heatmap_output = None
    if summary.get("heatmap") is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        heatmap_mod.generate(image_path, summary["heatmap"], ai_pct, tmp.name)
        heatmap_output = tmp.name
        _last_heatmap.append(tmp.name)

    # Indicators
    lines = []
    for r in results:
        for ind in r.indicators:
            lines.append(f"- **{r.analyzer}**: {ind}")
    indicators_md = "### Indicators\n" + "\n".join(lines) if lines else ""

    return verdict_html, df, heatmap_output, indicators_md


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="SlopDetector",
    theme=gr.themes.Soft(primary_hue="violet"),
    css=".verdict-box { min-height: 160px; }",
) as demo:
    gr.Markdown("# 🔍 SlopDetector\nUpload an image to estimate how likely it is to be AI-generated.")

    gr.HTML("""
<div style="background:#7c3a00;border:1px solid #e67e22;border-radius:8px;padding:12px 16px;
            font-family:system-ui,sans-serif;font-size:14px;color:#fde8c8;margin-bottom:8px;">
  ⚠️ <strong>Work in progress</strong> — accuracy varies by generator.
  Best results on DALL-E, Midjourney, and Stable Diffusion outputs.
  GPT-Image-2 detection is a known limitation under active development.
</div>""")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="filepath", label="Input image")
            analyze_btn = gr.Button("Analyse", variant="primary", size="lg")

        with gr.Column(scale=2):
            verdict_out = gr.HTML(elem_classes=["verdict-box"])
            breakdown_out = gr.Dataframe(
                headers=["Analyzer", "AI %", "Confidence", "Weight", "Top indicator"],
                label="Analyzer breakdown",
                interactive=False,
                wrap=True,
            )

    with gr.Row():
        heatmap_out = gr.Image(label="Spatial heatmap", show_label=True)
        indicators_out = gr.Markdown()

    analyze_btn.click(
        fn=analyze,
        inputs=img_input,
        outputs=[verdict_out, breakdown_out, heatmap_out, indicators_out],
    )
    img_input.change(
        fn=analyze,
        inputs=img_input,
        outputs=[verdict_out, breakdown_out, heatmap_out, indicators_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )

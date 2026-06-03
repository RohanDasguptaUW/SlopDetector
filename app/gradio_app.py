"""SlopDetector — Gradio web UI."""

import os
import sys
import tempfile
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parents[1]))

from ai_detector import ensemble
from ai_detector import heatmap as heatmap_mod
from ai_detector.analyzers.ela import ELAAnalyzer
from ai_detector.analyzers.spectral import SpectralAnalyzer
from ai_detector.analyzers.metadata import MetadataAnalyzer
from ai_detector.analyzers.noise import NoiseAnalyzer
from ai_detector.analyzers.ml import MLAnalyzer

_ANALYZERS = [ELAAnalyzer(), SpectralAnalyzer(), MetadataAnalyzer(), NoiseAnalyzer(), MLAnalyzer()]
_last_heatmap: list[str] = []

# ── Palette ───────────────────────────────────────────────────────────────────

def _score_color(pct: float) -> str:
    if pct >= 75: return "#f87171"   # red
    if pct >= 45: return "#fb923c"   # orange
    if pct >= 20: return "#fbbf24"   # yellow
    return "#4ade80"                  # green

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _ring(pct: float, color: str) -> str:
    """Conic-gradient ring with score inside."""
    return f"""
<div style="position:relative;width:148px;height:148px;border-radius:50%;flex-shrink:0;
            background:conic-gradient({color} {pct:.1f}%, #1e2030 0);
            display:flex;align-items:center;justify-content:center;">
  <div style="position:absolute;inset:0;border-radius:50%;
              background:conic-gradient({color} {pct:.1f}%, #1e2030 0);
              filter:blur(6px);opacity:0.35;"></div>
  <div style="position:relative;width:112px;height:112px;border-radius:50%;background:#12121e;
              display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;
              box-shadow:inset 0 2px 8px rgba(0,0,0,0.6);">
    <span style="font-size:30px;font-weight:800;color:{color};line-height:1;letter-spacing:-1px;">{pct:.0f}%</span>
    <span style="font-size:9px;font-weight:600;color:#4b5563;letter-spacing:0.12em;text-transform:uppercase;">AI Score</span>
  </div>
</div>"""


def _analyzer_row(name: str, pct: float, confidence: float, weight: float,
                  indicator: str, last: bool) -> str:
    color = _score_color(pct)
    border = "" if last else "border-bottom:1px solid #1e2030;"
    short = indicator[:90] + "…" if len(indicator) > 90 else indicator
    return f"""
<div style="display:grid;grid-template-columns:68px 1fr 46px 36px;
            align-items:center;gap:12px;padding:11px 0;{border}">
  <div style="font-size:10px;font-weight:700;letter-spacing:0.08em;
              text-transform:uppercase;color:#6b7280;">{name}</div>
  <div>
    <div style="height:5px;background:#1e2030;border-radius:3px;overflow:hidden;">
      <div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:3px;"></div>
    </div>
    <div style="font-size:11px;color:#4b5563;margin-top:5px;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{short}</div>
  </div>
  <div style="text-align:right;font-size:13px;font-weight:700;color:{color};">{pct:.0f}%</div>
  <div style="text-align:right;font-size:10px;color:#374151;font-weight:500;">{weight:.0%}</div>
</div>"""


def _indicator_items(results) -> str:
    items = ""
    for r in results:
        color = _score_color(r.ai_percentage)
        for ind in r.indicators:
            items += f"""
<div style="display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #1a1a28;align-items:flex-start;">
  <span style="font-size:10px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;
               color:{color};white-space:nowrap;padding-top:1px;min-width:64px;">{r.analyzer}</span>
  <span style="font-size:12px;color:#6b7280;line-height:1.5;">{ind}</span>
</div>"""
    return items


def _empty_state() -> str:
    return """
<div style="font-family:-apple-system,'Inter',system-ui,sans-serif;color:#e2e8f0;">
  <div style="background:#12121e;border:1px solid #1e2030;border-radius:16px;
              padding:60px 24px;text-align:center;">
    <div style="font-size:40px;margin-bottom:16px;opacity:0.3;">🔍</div>
    <div style="font-size:15px;font-weight:600;color:#374151;margin-bottom:6px;">No image uploaded</div>
    <div style="font-size:13px;color:#1f2937;">Upload an image to begin forensic analysis.</div>
  </div>
</div>"""


def _results_html(summary: dict, results: list) -> str:
    ai_pct = summary["ai_percentage"]
    confidence = summary["confidence"]
    verdict = summary["verdict"]
    weights = summary["weights_used"]
    color = _score_color(ai_pct)

    analyzer_rows = ""
    for i, r in enumerate(results):
        w = weights.get(r.analyzer, 0)
        ind = r.indicators[0] if r.indicators else "—"
        analyzer_rows += _analyzer_row(r.analyzer, r.ai_percentage, r.confidence, w, ind,
                                        last=(i == len(results) - 1))

    indicator_items = _indicator_items(results)

    return f"""
<div style="font-family:-apple-system,'Inter',system-ui,sans-serif;color:#e2e8f0;">

  <!-- Score card -->
  <div style="background:#12121e;border:1px solid #1e2030;border-radius:16px;
              padding:28px;margin-bottom:12px;display:flex;align-items:center;gap:28px;">
    {_ring(ai_pct, color)}
    <div style="flex:1;min-width:0;">
      <div style="font-size:22px;font-weight:700;color:#f1f5f9;margin-bottom:10px;
                  line-height:1.2;">{verdict}</div>
      <div style="display:inline-flex;align-items:center;gap:7px;
                  background:{color}18;border:1px solid {color}40;
                  border-radius:20px;padding:5px 13px;margin-bottom:14px;">
        <div style="width:7px;height:7px;border-radius:50%;background:{color};
                    box-shadow:0 0 6px {color};"></div>
        <span style="font-size:12px;color:{color};font-weight:600;">{confidence:.0%} confidence</span>
      </div>
      <div style="font-size:12px;color:#374151;">
        Ensemble of {len(results)} forensic signal{"s" if len(results) != 1 else ""}
      </div>
    </div>
  </div>

  <!-- Analyzer breakdown -->
  <div style="background:#12121e;border:1px solid #1e2030;border-radius:16px;
              padding:20px 24px;margin-bottom:12px;">
    <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                color:#374151;margin-bottom:4px;">Analyzer Breakdown</div>
    <div style="font-size:10px;color:#1f2937;margin-bottom:14px;">
      Score · Signal · AI % · Weight
    </div>
    {analyzer_rows}
  </div>

  <!-- Indicators -->
  <div style="background:#12121e;border:1px solid #1e2030;border-radius:16px;padding:20px 24px;">
    <div style="font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                color:#374151;margin-bottom:14px;">Forensic Indicators</div>
    {indicator_items}
  </div>

</div>"""


# ── Analyze function ──────────────────────────────────────────────────────────

def analyze(image_path: str | None):
    if _last_heatmap:
        try:
            os.unlink(_last_heatmap.pop())
        except OSError:
            pass

    if image_path is None:
        return _empty_state(), None

    results = []
    for analyzer in _ANALYZERS:
        try:
            results.append(analyzer.analyze(image_path))
        except Exception:
            pass

    if not results:
        return "<div style='color:#f87171;padding:24px;'>No analyzers could process this image.</div>", None

    summary = ensemble.combine(results)

    heatmap_output = None
    if summary.get("heatmap") is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        heatmap_mod.generate(image_path, summary["heatmap"], summary["ai_percentage"], tmp.name)
        heatmap_output = tmp.name
        _last_heatmap.append(tmp.name)

    return _results_html(summary, results), heatmap_output


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* ── Global ───────────────────────────────── */
body, .gradio-container, .main, .wrap { background: #0a0a10 !important; }
footer { display: none !important; }
.prose h1, .prose h2, .prose p { color: #e2e8f0; }

/* ── Upload area ──────────────────────────── */
#upload-col .wrap.svelte-1hnfib2,
#upload-col .wrap {
  background: #12121e !important;
  border: 1.5px dashed #2a2a3e !important;
  border-radius: 14px !important;
  min-height: 280px !important;
}
#upload-col .wrap:hover { border-color: #7c3aed !important; }
#upload-col label.float { color: #4b5563 !important; }

/* ── Analyse button ───────────────────────── */
#analyse-btn {
  background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
  border: none !important;
  border-radius: 10px !important;
  font-size: 15px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  height: 48px !important;
  box-shadow: 0 4px 20px rgba(124,58,237,0.35) !important;
  transition: box-shadow 0.2s ease !important;
}
#analyse-btn:hover {
  box-shadow: 0 6px 28px rgba(124,58,237,0.55) !important;
}

/* ── Results HTML panel ───────────────────── */
#results-panel { background: transparent !important; border: none !important; }
#results-panel > div { background: transparent !important; padding: 0 !important; }

/* ── Heatmap panel ────────────────────────── */
#heatmap-panel img { border-radius: 12px !important; }
#heatmap-panel { background: #12121e !important; border: 1px solid #1e2030 !important; border-radius: 16px !important; }
#heatmap-panel .wrap { background: transparent !important; border: none !important; }
#heatmap-panel label { color: #4b5563 !important; font-size: 10px !important;
  font-weight: 700 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }

/* ── Warning banner ───────────────────────── */
#warning-banner {
  background: #1c1408 !important;
  border: 1px solid #78350f !important;
  border-radius: 10px !important;
}
"""

# ── Layout ────────────────────────────────────────────────────────────────────

HEADER = """
<div style="font-family:-apple-system,'Inter',system-ui,sans-serif;
            padding:32px 0 20px;display:flex;align-items:center;gap:16px;">
  <div style="width:46px;height:46px;border-radius:12px;
              background:linear-gradient(135deg,#7c3aed,#4f46e5);
              display:flex;align-items:center;justify-content:center;
              font-size:22px;box-shadow:0 4px 16px rgba(124,58,237,0.4);">🔍</div>
  <div>
    <div style="font-size:22px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;">SlopDetector</div>
    <div style="font-size:12px;color:#4b5563;margin-top:1px;">AI image forensics</div>
  </div>
</div>"""

WARN = """
<div id="warning-banner" style="padding:11px 16px;font-size:12.5px;color:#d97706;
                                  font-family:-apple-system,'Inter',system-ui,sans-serif;
                                  margin-bottom:4px;">
  <strong style="color:#fbbf24;">⚠ Work in progress</strong> —
  accuracy varies by generator. Best results on DALL-E, Midjourney, and Stable Diffusion.
  GPT-Image-2 detection is a known limitation under active development.
</div>"""

with gr.Blocks(title="SlopDetector", css=CSS, theme=gr.themes.Base()) as demo:

    gr.HTML(HEADER)
    gr.HTML(WARN)

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_id="upload-col"):
            img_input = gr.Image(type="filepath", label="Drop image or click to upload",
                                 show_label=True)
            analyse_btn = gr.Button("Analyse Image", variant="primary",
                                    elem_id="analyse-btn", size="lg")

        with gr.Column(scale=2):
            results_out = gr.HTML(value=_empty_state(), elem_id="results-panel")

    with gr.Row():
        heatmap_out = gr.Image(label="Spatial Heatmap", show_label=True,
                               elem_id="heatmap-panel", visible=True)

    def _run(path):
        html, hm = analyze(path)
        return html, hm

    analyse_btn.click(fn=_run, inputs=img_input, outputs=[results_out, heatmap_out])
    img_input.change(fn=_run, inputs=img_input, outputs=[results_out, heatmap_out])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )

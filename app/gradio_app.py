"""SlopDetector — Gradio web UI."""

import math
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

GITHUB_URL = "https://github.com/RohanDasguptaUW/SlopDetector"

# ── Palette ───────────────────────────────────────────────────────────────────

def _score_color(pct: float) -> str:
    if pct >= 75: return "#f87171"
    if pct >= 45: return "#fb923c"
    if pct >= 20: return "#fbbf24"
    return "#4ade80"

def _score_label(pct: float) -> str:
    if pct >= 75: return "AI GENERATED"
    if pct >= 45: return "LIKELY AI"
    if pct >= 20: return "POSSIBLY AI"
    return "LIKELY REAL"

# ── HTML helpers ──────────────────────────────────────────────────────────────

def _gauge(pct: float, color: str) -> str:
    """SVG half-circle gauge — more impressive than a full ring."""
    r = 88
    total_arc = math.pi * r          # half-circle arc ≈ 276.5
    filled = (pct / 100) * total_arc
    label = _score_label(pct)
    return f"""
<div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;">
  <svg viewBox="0 0 200 115" width="200" height="115"
       style="overflow:visible;filter:drop-shadow(0 0 16px {color}40);">
    <!-- Track -->
    <path d="M 12 100 A 88 88 0 0 1 188 100"
          stroke="#16162a" stroke-width="14" fill="none" stroke-linecap="round"/>
    <!-- Filled arc -->
    <path d="M 12 100 A 88 88 0 0 1 188 100"
          stroke="{color}" stroke-width="14" fill="none" stroke-linecap="round"
          stroke-dasharray="{filled:.2f} {total_arc:.2f}"
          style="filter:drop-shadow(0 0 6px {color});"/>
    <!-- Score -->
    <text x="100" y="76" text-anchor="middle"
          fill="{color}" font-size="40" font-weight="800"
          font-family="-apple-system,Inter,system-ui,sans-serif"
          letter-spacing="-1">{pct:.0f}%</text>
    <!-- Sub-label -->
    <text x="100" y="96" text-anchor="middle"
          fill="#2e2e50" font-size="9" font-weight="700"
          font-family="-apple-system,Inter,system-ui,sans-serif"
          letter-spacing="2.5">AI SCORE</text>
  </svg>
  <div style="font-size:9px;font-weight:700;letter-spacing:0.18em;
              color:{color};text-transform:uppercase;margin-top:-4px;
              opacity:0.9;">{label}</div>
</div>"""


def _analyzer_row(name: str, pct: float, confidence: float, weight: float,
                  indicator: str, last: bool) -> str:
    color = _score_color(pct)
    border = "" if last else "border-bottom:1px solid #10101e;"
    short = indicator[:100] + "…" if len(indicator) > 100 else indicator
    bar_w = max(2, pct)
    return f"""
<div style="padding:11px 0;{border}">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:7px;">
    <div style="display:flex;align-items:center;gap:7px;">
      <div style="width:5px;height:5px;border-radius:50%;background:{color};
                  box-shadow:0 0 5px {color};flex-shrink:0;"></div>
      <span style="font-size:10px;font-weight:700;letter-spacing:0.1em;
                   text-transform:uppercase;color:#9ca3af;">{name}</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-size:9px;color:#2e2e50;">conf {confidence:.0%}</span>
      <span style="font-size:9px;color:#2e2e50;">wt {weight:.0%}</span>
      <span style="font-size:15px;font-weight:800;color:{color};
                   min-width:36px;text-align:right;">{pct:.0f}%</span>
    </div>
  </div>
  <div style="height:5px;background:#10101e;border-radius:3px;overflow:hidden;margin-bottom:6px;">
    <div style="width:{bar_w:.0f}%;height:100%;
                background:linear-gradient(90deg,{color}70,{color});
                border-radius:3px;"></div>
  </div>
  <div style="font-size:11px;color:#4b5563;line-height:1.4;padding-left:12px;">{short}</div>
</div>"""


def _indicator_items(results) -> str:
    items = ""
    for r in results:
        color = _score_color(r.ai_percentage)
        for ind in r.indicators:
            items += f"""
<div style="display:flex;gap:9px;padding:7px 0;border-bottom:1px solid #0c0c1c;align-items:flex-start;">
  <div style="width:5px;height:5px;border-radius:50%;background:{color};flex-shrink:0;
              margin-top:5px;box-shadow:0 0 4px {color};"></div>
  <span style="font-size:9px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;
               color:{color};white-space:nowrap;min-width:54px;">{r.analyzer}</span>
  <span style="font-size:11px;color:#6b7280;line-height:1.5;">{ind}</span>
</div>"""
    return items


def _empty_state() -> str:
    return """
<div style="font-family:-apple-system,'Inter',system-ui,sans-serif;">
  <div style="background:#0a0a18;border:1px dashed #16162a;border-radius:14px;
              padding:72px 24px;text-align:center;">
    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#20203a"
         stroke-width="1.5" style="margin:0 auto 14px;display:block;">
      <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
    </svg>
    <div style="font-size:14px;font-weight:600;color:#20203a;margin-bottom:5px;">No image uploaded</div>
    <div style="font-size:12px;color:#16162a;">Upload an image to begin forensic analysis.</div>
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
  <div style="background:#0a0a18;border:1px solid #16162a;border-radius:14px;
              padding:22px 24px;margin-bottom:10px;display:flex;align-items:center;gap:18px;">
    {_gauge(ai_pct, color)}
    <div style="flex:1;min-width:0;">
      <div style="font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:10px;line-height:1.2;">{verdict}</div>
      <div style="display:inline-flex;align-items:center;gap:7px;
                  background:{color}12;border:1px solid {color}28;
                  border-radius:20px;padding:4px 12px;margin-bottom:12px;">
        <div style="width:6px;height:6px;border-radius:50%;background:{color};
                    box-shadow:0 0 8px {color};animation:pulse 2s infinite;"></div>
        <span style="font-size:11px;color:{color};font-weight:600;">{confidence:.0%} confidence</span>
      </div>
      <div style="font-size:11px;color:#24243c;">
        {len(results)} forensic signal{"s" if len(results) != 1 else ""} · weighted ensemble
      </div>
    </div>
  </div>

  <!-- Analyzer breakdown -->
  <div style="background:#0a0a18;border:1px solid #16162a;border-radius:14px;
              padding:16px 20px;margin-bottom:10px;">
    <div style="font-size:9px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
                color:#2e2e50;margin-bottom:12px;">Analyzer Breakdown</div>
    {analyzer_rows}
  </div>

  <!-- Indicators -->
  <div style="background:#0a0a18;border:1px solid #16162a;border-radius:14px;padding:16px 20px;">
    <div style="font-size:9px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
                color:#2e2e50;margin-bottom:10px;">Forensic Indicators</div>
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
        return _empty_state(), gr.update(visible=False), gr.update(visible=False)

    results = []
    for analyzer in _ANALYZERS:
        try:
            results.append(analyzer.analyze(image_path))
        except Exception:
            pass

    if not results:
        err = "<div style='color:#f87171;padding:24px;font-family:system-ui;'>No analyzers could process this image.</div>"
        return err, gr.update(visible=False), gr.update(visible=False)

    summary = ensemble.combine(results)

    heatmap_path = None
    if summary.get("heatmap") is not None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        heatmap_mod.generate(image_path, summary["heatmap"], summary["ai_percentage"], tmp.name)
        heatmap_path = tmp.name
        _last_heatmap.append(tmp.name)

    has_heatmap = heatmap_path is not None
    return (
        _results_html(summary, results),
        gr.update(value=image_path, visible=has_heatmap),
        gr.update(value=heatmap_path, visible=has_heatmap),
    )


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
/* ── Global ───────────────────────────────── */
body, .gradio-container { background: #07070f !important; }
footer { display: none !important; }
.prose h1, .prose h2, .prose p { color: #e2e8f0; }

/* ── Grid texture on body ─────────────────── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(124,58,237,0.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(124,58,237,0.045) 1px, transparent 1px);
  background-size: 50px 50px;
  pointer-events: none;
  z-index: 0;
}

/* ── Scanline overlay ─────────────────────── */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 3px,
    rgba(0,0,0,0.065) 3px, rgba(0,0,0,0.065) 4px
  );
  pointer-events: none;
  z-index: 0;
}

/* ── Strip all panel/block chrome ────────── */
.block, .form, .gap, [class*="block"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* ── Upload component — kill every white box ─
   Strategy: make all divs/spans dark, then
   restore transparent for actual media.       */
#upload-col div,
#upload-col span,
#upload-col section,
#upload-col label {
  background-color: #0a0a18 !important;
}
#upload-col img,
#upload-col canvas,
#upload-col svg,
#upload-col video { background-color: transparent !important; }

/* Drop zone */
#upload-col .wrap,
#upload-col [class*="wrap"] {
  background: #0a0a18 !important;
  border: 1.5px dashed #1e1e38 !important;
  border-radius: 14px !important;
  min-height: 280px !important;
  color: #2e2e50 !important;
}
#upload-col .wrap:hover,
#upload-col [class*="wrap"]:hover {
  border-color: #7c3aed !important;
  box-shadow: inset 0 0 56px rgba(124,58,237,0.07) !important;
}
/* Upload icon/placeholder text */
#upload-col .wrap svg  { stroke: #1e1e38 !important; fill: none !important; }
#upload-col .wrap span { color: #2e2e50 !important; }
/* Uploaded image */
#upload-col img { border-radius: 10px !important; background: transparent !important; }
/* Clear / edit / camera buttons */
#upload-col button {
  background: #12122a !important;
  border: 1px solid #1e1e38 !important;
  color: #4b5563 !important;
  border-radius: 7px !important;
}
#upload-col button:hover {
  background: #1a1a38 !important;
  border-color: #7c3aed !important;
}

/* ── Analyse button ───────────────────────── */
#analyse-btn {
  background: linear-gradient(135deg, #7c3aed, #5b21b6) !important;
  border: none !important;
  border-radius: 10px !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  height: 48px !important;
  box-shadow: 0 4px 26px rgba(124,58,237,0.45),
              inset 0 1px 0 rgba(255,255,255,0.08) !important;
  transition: box-shadow 0.15s ease, transform 0.15s ease !important;
}
#analyse-btn:hover {
  box-shadow: 0 6px 34px rgba(124,58,237,0.65),
              inset 0 1px 0 rgba(255,255,255,0.12) !important;
  transform: translateY(-1px) !important;
}
#analyse-btn:active { transform: translateY(0) !important; }

/* ── Results HTML panel ───────────────────── */
#results-panel,
#results-panel > div {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

/* ── Side-by-side comparison panels ──────── */
#orig-panel,
#heatmap-panel {
  background: #0a0a18 !important;
  border: 1px solid #16162a !important;
  border-radius: 14px !important;
}
#orig-panel div,
#heatmap-panel div,
#orig-panel span,
#heatmap-panel span { background-color: transparent !important; }
#orig-panel img, #heatmap-panel img {
  border-radius: 10px !important;
  background: transparent !important;
}
#orig-panel .wrap, #heatmap-panel .wrap,
#orig-panel [class*="wrap"], #heatmap-panel [class*="wrap"] {
  background: transparent !important;
  border: none !important;
  min-height: 180px !important;
}
#orig-panel label, #heatmap-panel label {
  color: #2e2e50 !important;
  font-size: 9px !important;
  font-weight: 700 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
}

/* ── Warning banner ───────────────────────── */
#warning-banner {
  background: #130c00 !important;
  border: 1px solid #78350f !important;
  border-radius: 10px !important;
}

/* ── Footer ───────────────────────────────── */
#app-footer > div { background: transparent !important; padding: 0 !important; }
#app-footer,
#app-footer * { background-color: transparent !important; }

/* ── Animations ───────────────────────────── */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.35} }
"""

# ── Layout constants ──────────────────────────────────────────────────────────

HEADER = """
<div style="font-family:-apple-system,'Inter',system-ui,sans-serif;
            padding:24px 0 14px;display:flex;align-items:center;gap:14px;">
  <div style="width:42px;height:42px;border-radius:10px;flex-shrink:0;
              background:linear-gradient(135deg,#7c3aed,#4f46e5);
              display:flex;align-items:center;justify-content:center;
              font-size:19px;
              box-shadow:0 4px 22px rgba(124,58,237,0.5),inset 0 1px 0 rgba(255,255,255,0.15);">🔍</div>
  <div>
    <div style="font-size:20px;font-weight:800;color:#f1f5f9;letter-spacing:-0.5px;">SlopDetector</div>
    <div style="font-size:9px;color:#24243c;margin-top:3px;letter-spacing:0.12em;text-transform:uppercase;">
      AI Image Forensics
    </div>
  </div>
</div>"""

WARN = """
<div id="warning-banner" style="padding:9px 15px;font-size:11.5px;color:#d97706;
                                  font-family:-apple-system,'Inter',system-ui,sans-serif;
                                  margin-bottom:8px;">
  <strong style="color:#fbbf24;">⚠ Work in progress</strong> —
  accuracy varies by generator. Best results on DALL-E, Midjourney, and Stable Diffusion outputs.
  GPT-Image-2 detection is a known limitation under active development.
</div>"""

FOOTER = f"""
<div id="app-footer" style="font-family:-apple-system,'Inter',system-ui,sans-serif;
                             margin-top:28px;padding:22px 0 10px;
                             border-top:1px solid #10101e;">

  <div style="font-size:9px;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
              color:#24243c;margin-bottom:12px;">How It Works</div>

  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(168px,1fr));gap:8px;margin-bottom:18px;">

    <div style="background:#0a0a18;border:1px solid #12122a;border-radius:10px;padding:11px 13px;">
      <div style="font-size:9px;font-weight:700;color:#6d28d9;letter-spacing:0.1em;
                  text-transform:uppercase;margin-bottom:4px;">ELA</div>
      <div style="font-size:10.5px;color:#374151;line-height:1.5;">
        Error Level Analysis detects compression artifacts inconsistent with a real photo's history.
      </div>
    </div>

    <div style="background:#0a0a18;border:1px solid #12122a;border-radius:10px;padding:11px 13px;">
      <div style="font-size:9px;font-weight:700;color:#6d28d9;letter-spacing:0.1em;
                  text-transform:uppercase;margin-bottom:4px;">Spectral</div>
      <div style="font-size:10.5px;color:#374151;line-height:1.5;">
        Frequency-domain analysis reveals unnatural spectral patterns left by generative models.
      </div>
    </div>

    <div style="background:#0a0a18;border:1px solid #12122a;border-radius:10px;padding:11px 13px;">
      <div style="font-size:9px;font-weight:700;color:#6d28d9;letter-spacing:0.1em;
                  text-transform:uppercase;margin-bottom:4px;">Metadata</div>
      <div style="font-size:10.5px;color:#374151;line-height:1.5;">
        EXIF and creation metadata differ between AI generators and real cameras with optics.
      </div>
    </div>

    <div style="background:#0a0a18;border:1px solid #12122a;border-radius:10px;padding:11px 13px;">
      <div style="font-size:9px;font-weight:700;color:#6d28d9;letter-spacing:0.1em;
                  text-transform:uppercase;margin-bottom:4px;">Noise</div>
      <div style="font-size:10.5px;color:#374151;line-height:1.5;">
        Pixel-level noise distribution — AI images exhibit unnaturally uniform or structured noise.
      </div>
    </div>

    <div style="background:#0a0a18;border:1px solid #12122a;border-radius:10px;padding:11px 13px;">
      <div style="font-size:9px;font-weight:700;color:#6d28d9;letter-spacing:0.1em;
                  text-transform:uppercase;margin-bottom:4px;">ML</div>
      <div style="font-size:10.5px;color:#374151;line-height:1.5;">
        ResNet50 fine-tuned on real vs AI image datasets for direct binary classification.
      </div>
    </div>

  </div>

  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;">
    <span style="font-size:10px;color:#16162a;">SlopDetector · open-source AI image forensics</span>
    <a href="{GITHUB_URL}" target="_blank" rel="noopener"
       style="font-size:10px;color:#2e2e50;text-decoration:none;
              display:flex;align-items:center;gap:5px;">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
        <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483
                 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466
                 -1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53
                 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951
                 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026
                 A9.564 9.564 0 0 1 12 6.844a9.59 9.59 0 0 1 2.504.337c1.909-1.296 2.747-1.027
                 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848
                 -2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747
                 0 .268.18.58.688.482A10.02 10.02 0 0 0 22 12.017C22 6.484 17.522 2 12 2z"/>
      </svg>
      GitHub
    </a>
  </div>

</div>"""


# ── Build UI ──────────────────────────────────────────────────────────────────

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

    # Side-by-side comparison row — hidden until a heatmap is generated
    with gr.Row(equal_height=True):
        orig_out = gr.Image(label="Original", show_label=True, interactive=False,
                            elem_id="orig-panel", visible=False)
        heatmap_out = gr.Image(label="Spatial Heatmap", show_label=True, interactive=False,
                               elem_id="heatmap-panel", visible=False)

    gr.HTML(FOOTER)

    analyse_btn.click(fn=analyze, inputs=img_input,
                      outputs=[results_out, orig_out, heatmap_out])
    img_input.change(fn=analyze, inputs=img_input,
                     outputs=[results_out, orig_out, heatmap_out])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )

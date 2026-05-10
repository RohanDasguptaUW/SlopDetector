"""Self-contained HTML report generator."""

import base64
import json
from pathlib import Path


def _b64_embed(path: str, mime: str = "image/jpeg") -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def generate(
    summary: dict,
    image_path: str,
    heatmap_path: str | None,
    output_path: str,
) -> str:
    """Generate self-contained HTML report and save to output_path."""
    ai_pct = summary["ai_percentage"]
    confidence = summary["confidence"]
    verdict = summary["verdict"]
    per_analyzer = summary.get("per_analyzer", {})
    weights = summary.get("weights_used", {})

    # Colour for verdict badge
    if ai_pct >= 75:
        badge_colour = "#e74c3c"
    elif ai_pct >= 45:
        badge_colour = "#e67e22"
    elif ai_pct >= 20:
        badge_colour = "#f1c40f"
    else:
        badge_colour = "#27ae60"

    # Embed images
    img_b64 = _b64_embed(image_path, "image/jpeg")
    heatmap_section = ""
    if heatmap_path and Path(heatmap_path).exists():
        hm_b64 = _b64_embed(heatmap_path, "image/png")
        heatmap_section = f'<img src="{hm_b64}" style="max-width:100%;border-radius:6px;" alt="Heatmap">'

    # Per-analyzer rows
    analyzer_rows = ""
    for name, res in per_analyzer.items():
        pct = res["ai_percentage"]
        bar_w = int(pct)
        bar_col = "#e74c3c" if pct >= 70 else "#e67e22" if pct >= 40 else "#27ae60"
        w = weights.get(name, 0)
        indicators_str = "; ".join(res.get("indicators", []))
        analyzer_rows += f"""
        <tr>
          <td style="padding:8px;font-weight:600;text-transform:capitalize;">{name}</td>
          <td style="padding:8px;">
            <div style="background:#555;border-radius:4px;height:14px;width:200px;">
              <div style="background:{bar_col};width:{bar_w}%;height:100%;border-radius:4px;"></div>
            </div>
            <span style="font-size:12px;">{pct:.1f}%</span>
          </td>
          <td style="padding:8px;">{res['confidence']:.2f}</td>
          <td style="padding:8px;font-size:11px;">{w:.0%}</td>
          <td style="padding:8px;font-size:11px;color:#aaa;">{indicators_str}</td>
        </tr>"""

    # All indicators
    all_indicators = []
    for res in per_analyzer.values():
        all_indicators.extend(res.get("indicators", []))
    indicator_list = "".join(f"<li>{i}</li>" for i in all_indicators)

    raw_json = json.dumps(summary, indent=2, default=str)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SlopDetector Report</title>
  <style>
    body {{ background:#1a1a2e; color:#e0e0e0; font-family:system-ui,sans-serif; margin:0; padding:20px; }}
    .container {{ max-width:1100px; margin:auto; }}
    h1 {{ color:#a29bfe; }} h2 {{ color:#74b9ff; border-bottom:1px solid #333; padding-bottom:6px; }}
    .card {{ background:#16213e; border-radius:10px; padding:20px; margin-bottom:20px; }}
    .big-num {{ font-size:72px; font-weight:900; color:{badge_colour}; }}
    .badge {{ display:inline-block; background:{badge_colour}; color:#fff; padding:6px 16px;
               border-radius:20px; font-weight:700; font-size:16px; margin-top:8px; }}
    table {{ width:100%; border-collapse:collapse; }}
    tr:nth-child(even) {{ background:#0d1b2a; }}
    th {{ background:#0f3460; padding:10px; text-align:left; }}
    details summary {{ cursor:pointer; color:#74b9ff; font-weight:600; margin-top:10px; }}
    pre {{ background:#0d1b2a; padding:15px; border-radius:6px; overflow-x:auto;
            font-size:12px; max-height:400px; overflow-y:auto; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
    @media(max-width:700px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
<div class="container">
  <h1>🔍 SlopDetector Report</h1>

  <div class="card">
    <div class="grid">
      <div>
        <div>AI-Generated Probability</div>
        <div class="big-num">{ai_pct:.0f}%</div>
        <div class="badge">{verdict}</div>
        <p style="margin-top:12px;color:#aaa;">Confidence: {confidence:.0%}</p>
      </div>
      <div>
        <img src="{img_b64}" style="max-width:100%;max-height:300px;border-radius:6px;object-fit:contain;" alt="Input image">
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Analyzer Breakdown</h2>
    <table>
      <thead>
        <tr>
          <th>Analyzer</th><th>AI Score</th><th>Confidence</th><th>Weight</th><th>Indicators</th>
        </tr>
      </thead>
      <tbody>{analyzer_rows}</tbody>
    </table>
  </div>

  {f'<div class="card"><h2>Spatial Heatmap</h2>{heatmap_section}</div>' if heatmap_section else ''}

  <div class="card">
    <h2>All Indicators</h2>
    <ul style="font-size:13px;line-height:1.8;">{indicator_list}</ul>
  </div>

  <div class="card">
    <details>
      <summary>Raw JSON Data</summary>
      <pre>{raw_json}</pre>
    </details>
  </div>

  <p style="color:#555;font-size:12px;text-align:center;">
    Generated by SlopDetector · Results are probabilistic and should not be used as sole evidence
  </p>
</div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

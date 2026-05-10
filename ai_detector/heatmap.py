"""Heatmap visualisation: original + false-colour overlay side-by-side."""

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _false_colour(heatmap: np.ndarray) -> np.ndarray:
    """Map 0–1 float heatmap to blue→red RGB."""
    h = np.clip(heatmap, 0.0, 1.0)
    r = h
    g = np.zeros_like(h)
    b = 1.0 - h
    rgb = np.stack([r, g, b], axis=2)
    return (rgb * 255).astype(np.uint8)


def _make_colorbar(height: int, width: int = 30) -> np.ndarray:
    """Vertical colour bar from blue (0) to red (1)."""
    grad = np.linspace(1.0, 0.0, height)
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:, :, 0] = (grad * 255).astype(np.uint8)[:, None]
    bar[:, :, 2] = ((1 - grad) * 255).astype(np.uint8)[:, None]
    return bar


def generate(image_path: str, heatmap: np.ndarray, ai_percentage: float, output_path: str) -> str:
    """Generate side-by-side PNG and save to output_path. Returns output_path."""
    original = Image.open(image_path).convert("RGB")
    W, H = original.size

    # Resize heatmap to image dimensions if needed
    if heatmap.shape != (H, W):
        pil_hm = Image.fromarray((heatmap * 255).astype(np.uint8))
        pil_hm = pil_hm.resize((W, H), Image.BICUBIC)
        heatmap_resized = np.array(pil_hm, dtype=np.float32) / 255.0
    else:
        heatmap_resized = heatmap

    # False-colour overlay
    fc = _false_colour(heatmap_resized)
    fc_img = Image.fromarray(fc)
    overlay = Image.blend(original, fc_img, alpha=0.45)

    # Colorbar
    cb_w = 30
    cb_arr = _make_colorbar(H, cb_w)
    cb_img = Image.fromarray(cb_arr)

    # Label bar at top
    label_h = 40
    total_w = W + W + cb_w + 20  # left pad, original, gap, overlay, colorbar
    canvas_h = H + label_h
    canvas = Image.new("RGB", (total_w, canvas_h), color=(30, 30, 30))

    # Paste panels
    canvas.paste(original, (0, label_h))
    canvas.paste(overlay, (W + 10, label_h))
    canvas.paste(cb_img, (W + 10 + W + 5, label_h))

    # Add labels
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    draw.text((5, 10), "Original", fill=(220, 220, 220), font=font)
    overlay_label = f"AI Heatmap — {ai_percentage:.1f}% AI"
    draw.text((W + 15, 10), overlay_label, fill=(220, 220, 220), font=font)

    # Colorbar tick labels
    draw.text((W + W + 17, label_h), "100%", fill=(200, 200, 200), font=small_font)
    draw.text((W + W + 17, label_h + H // 2 - 6), " 50%", fill=(200, 200, 200), font=small_font)
    draw.text((W + W + 17, label_h + H - 16), "  0%", fill=(200, 200, 200), font=small_font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path

"""One-off script to render the 1200x630 OG card for social previews.

Run once: `python scripts/generate_og_card.py`. Commit the resulting PNG.
Regenerate only when branding or tagline changes.
"""
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1200, 630
BG = (15, 17, 23)          # --bg (dark navy)
ACCENT = (108, 114, 255)   # --accent (indigo)
ACCENT2 = (78, 205, 196)   # --accent2 (teal)
GREEN = (34, 197, 94)
YELLOW = (245, 158, 11)
TEXT = (228, 231, 240)
TEXT2 = (139, 144, 165)


def _load_font(size: int, bold: bool = False):
    """Best-effort font loader. Falls back to default if system fonts missing."""
    candidates_bold = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSDisplay-Bold.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
    ]
    candidates_regular = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "Arial.ttf",
    ]
    for path in (candidates_bold if bold else candidates_regular):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def main():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # Top-left accent stripe
    draw.rectangle([(0, 0), (16, HEIGHT)], fill=ACCENT)

    # Bar-chart motif (top-right, subtle)
    bars = [
        (WIDTH - 260, 80, 50, 100, ACCENT),
        (WIDTH - 200, 80, 50, 160, ACCENT2),
        (WIDTH - 140, 80, 50, 220, GREEN),
        (WIDTH - 80, 80, 50, 180, YELLOW),
    ]
    for x, y, w, h, color in bars:
        draw.rectangle([(x, y + (240 - h)), (x + w, y + 240)], fill=color)

    # Title
    title = "BenchPress"
    title_font = _load_font(120, bold=True)
    title_w = _text_width(draw, title, title_font)
    draw.text(((WIDTH - title_w) // 2, 210), title, font=title_font, fill=TEXT)

    # Tagline
    tagline = "LLM Evaluation & Causal Reasoning Benchmark"
    tag_font = _load_font(42, bold=False)
    tag_w = _text_width(draw, tagline, tag_font)
    draw.text(((WIDTH - tag_w) // 2, 360), tagline, font=tag_font, fill=ACCENT2)

    # Stats line
    stats = "50+ models \u00b7 100 causal questions \u00b7 20 bundles \u00b7 5 variants"
    stats_font = _load_font(28, bold=False)
    stats_w = _text_width(draw, stats, stats_font)
    draw.text(((WIDTH - stats_w) // 2, 440), stats, font=stats_font, fill=TEXT2)

    # URL footer
    url = "mark-allwyn.github.io/BenchPress"
    url_font = _load_font(26, bold=False)
    url_w = _text_width(draw, url, url_font)
    draw.text(((WIDTH - url_w) // 2, 540), url, font=url_font, fill=TEXT2)

    out = Path("docs/og-card.png")
    out.parent.mkdir(exist_ok=True)
    img.save(out, "PNG", optimize=True)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB, {WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()

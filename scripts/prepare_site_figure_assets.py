#!/usr/bin/env python3
"""Copy and theme figure assets for the public site (dark UI)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "assets"
DST = ROOT / "site" / "assets"

# Site token --bg / --panel
SITE_BG = (8, 8, 10)
SITE_PANEL = (17, 17, 20)

COPY_AS_IS = ("public_exactkv_one_page_summary.png",)
DARKEN_CHARTS = (
    "exp035_first_divergence_histogram.png",
    "exp035_category_heatmap.png",
)


def _darken_matplotlib_png(src: Path, dst: Path) -> None:
    """Invert light-theme matplotlib exports for dark landing page."""
    from PIL import Image, ImageEnhance, ImageOps

    im = Image.open(src).convert("RGB")
    im = ImageOps.invert(im)
    try:
        import numpy as np

        arr = np.asarray(im).copy()
        mask = (arr[:, :, 0] < 28) & (arr[:, :, 1] < 28) & (arr[:, :, 2] < 28)
        arr[mask] = SITE_BG
        im = Image.fromarray(arr)
    except ImportError:
        px = im.load()
        w, h = im.size
        for y in range(h):
            for x in range(w):
                r, g, b = px[x, y]
                if r < 28 and g < 28 and b < 28:
                    px[x, y] = SITE_BG
    im = ImageEnhance.Contrast(im).enhance(1.08)
    im = ImageEnhance.Brightness(im).enhance(0.95)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, optimize=True)


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    for name in COPY_AS_IS:
        src = SRC / name
        if not src.is_file():
            raise FileNotFoundError(src)
        (DST / name).write_bytes(src.read_bytes())
        print(f"copied {name}")

    for name in DARKEN_CHARTS:
        src = SRC / name
        if not src.is_file():
            raise FileNotFoundError(src)
        _darken_matplotlib_png(src, DST / name)
        print(f"dark-themed {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

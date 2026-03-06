#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "numpy",
# ]
# ///
"""
Image processor: crop to WxH px, reduce to 6 colors via Floyd-Steinberg dithering.

Usage:
  uv run process_images.py                  # uses config.json or built-in defaults
  uv run process_images.py <folder_path>    # overrides photo_dir from config/defaults

Config (optional config.json next to this script):
  {
    "photo_dir": "./photos",
    "extensions": [".jpg", ".jpeg", ".png"],
    "width": 1200,
    "height": 1600
  }
"""

import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np

# ── 6-color palette (feel free to customize) ──────────────────────────────────
PALETTE_6 = np.array([
    [0,   0,   0  ],  # black
    [255, 255, 255],  # white
    [255, 0,   0  ],  # red
    [0,   255, 0  ],  # green
    [0,   0,   255],  # blue
    [255, 255, 0  ],  # yellow
], dtype=np.float32)

# ── Built-in defaults (used when config.json is absent / key is missing) ──────
DEFAULTS = {
    "photo_dir":  "./photos",
    "extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"],
    "width":      1200,
    "height":     1600,
}

CONFIG_FILE = Path(__file__).parent / "config.json"


def load_config() -> dict:
    """Merge config.json (if present) over built-in defaults. All keys optional."""
    cfg = DEFAULTS.copy()
    if CONFIG_FILE.exists():
        try:
            with CONFIG_FILE.open() as f:
                overrides = json.load(f)
            cfg.update({k: v for k, v in overrides.items() if k in DEFAULTS})
            print(f"⚙️  Config loaded from {CONFIG_FILE.name}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  Could not read config.json ({e}), using defaults.")
    else:
        print(f"ℹ️  No config.json found, using built-in defaults.")
    return cfg


def smart_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale then center-crop to exactly target_w × target_h."""
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def nearest_color(pixel: np.ndarray, palette: np.ndarray) -> tuple[int, np.ndarray]:
    """Return index + RGB of the closest palette color (Euclidean in RGB)."""
    diffs = palette - pixel
    dists = (diffs ** 2).sum(axis=1)
    idx = int(np.argmin(dists))
    return idx, palette[idx]


def floyd_steinberg_dither(img: Image.Image, palette: np.ndarray) -> Image.Image:
    """Apply Floyd-Steinberg dithering, mapping every pixel to one of 6 palette colors."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            old_pixel = arr[y, x].copy()
            _, new_pixel = nearest_color(old_pixel, palette)
            out[y, x] = new_pixel.astype(np.uint8)
            quant_error = old_pixel - new_pixel

            # Distribute error to neighbours (Floyd-Steinberg weights)
            if x + 1 < w:
                arr[y,     x + 1] += quant_error * (7 / 16)
            if y + 1 < h:
                if x - 1 >= 0:
                    arr[y + 1, x - 1] += quant_error * (3 / 16)
                arr[y + 1, x    ] += quant_error * (5 / 16)
                if x + 1 < w:
                    arr[y + 1, x + 1] += quant_error * (1 / 16)

        # Progress bar
        if y % 100 == 0:
            pct = int(y / h * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  dithering [{bar}] {pct}%", end="", flush=True)

    print(f"\r  dithering [{'█'*20}] 100%")
    return Image.fromarray(out, "RGB")


def process_folder(folder: str, target_w: int, target_h: int, supported: set) -> None:
    src_dir = Path(folder).resolve()
    if not src_dir.is_dir():
        print(f"❌  Not a directory: {src_dir}")
        sys.exit(1)

    out_dir = src_dir / "img"
    out_dir.mkdir(exist_ok=True)
    print(f"📂  Source : {src_dir}")
    print(f"📁  Output : {out_dir}")
    print(f"📐  Size   : {target_w}×{target_h}")
    print(f"🔍  Exts   : {', '.join(sorted(supported))}\n")

    images = [p for p in src_dir.iterdir() if p.suffix.lower() in supported]
    if not images:
        print("⚠️  No supported images found.")
        return

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")

        try:
            with Image.open(img_path) as img:
                # 1. Crop to target size
                print(f"  cropping  {img.size} → ({target_w}×{target_h})")
                cropped = smart_crop(img, target_w, target_h)

                # 2. Floyd-Steinberg dither to 6 colors
                dithered = floyd_steinberg_dither(cropped, PALETTE_6)

                # 3. Save
                out_name = img_path.stem + "_dithered.png"
                out_path = out_dir / out_name
                dithered.save(out_path, "PNG", optimize=True)
                kb = out_path.stat().st_size // 1024
                print(f"  ✅ saved  → {out_name}  ({kb} KB)\n")

        except Exception as e:
            print(f"  ⚠️  Skipped: {e}\n")

    print(f"Done! {len(images)} image(s) processed → {out_dir}")


if __name__ == "__main__":
    cfg = load_config()

    # CLI arg overrides photo_dir from config/defaults
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["photo_dir"]

    process_folder(
        folder=folder,
        target_w=cfg["width"],
        target_h=cfg["height"],
        supported=set(cfg["extensions"]),
    )   process_folder(sys.argv[1])



#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "Pillow",
#   "numpy",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""
Image processor: crop → adaptive palette → dither to N colors.

Usage:
  uv run process_images.py                  # uses config.json or built-in defaults
  uv run process_images.py <folder_path>    # overrides photo_dir

Dithering algorithms (set via config "dither_algorithm"):
  floyd_steinberg   – classic, fast, slight worm artifacts
  jarvis            – wider 12-neighbor spread, smoother gradients
  ostromoukhov      – variable per-pixel weights, near-optimal classic
  blue_noise        – most natural, zero repeating patterns  (default)

Palette modes (set via config "palette_mode"):
  fixed             – use hardcoded colors from "palette" in config
  adaptive          – k-means per image, best quality             (default)
"""

import sys
import json
import math
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# ── Built-in defaults ──────────────────────────────────────────────────────────
DEFAULTS: dict = {
    "photo_dir":        "./photos",
    "extensions":       [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"],
    "width":            1200,
    "height":           1600,
    "num_colors":       6,
    "dither_algorithm": "blue_noise",   # floyd_steinberg | jarvis | ostromoukhov | blue_noise
    "palette_mode":     "adaptive",     # adaptive | fixed
    "palette": [                        # used only when palette_mode = "fixed"
        [0,   0,   0  ],
        [255, 255, 255],
        [255, 0,   0  ],
        [0,   255, 0  ],
        [0,   0,   255],
        [255, 255, 0  ],
    ],
}

CONFIG_FILE = Path(__file__).parent / "config.json"


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
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
        print("ℹ️  No config.json found, using built-in defaults.")
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Gamma-correct colour space helpers
# ══════════════════════════════════════════════════════════════════════════════

def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB [0-255] float32 → linear [0-1] (proper perceptual gamma)."""
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)

def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear [0-1] → sRGB [0-255] uint8."""
    c = np.clip(c, 0.0, 1.0)
    out = np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1.0 / 2.4)) - 0.055)
    return (out * 255.0).round().astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Palette building
# ══════════════════════════════════════════════════════════════════════════════

def build_adaptive_palette(img: Image.Image, n: int) -> np.ndarray:
    """K-means in linear RGB to find the n most representative colors."""
    print(f"  🎨 building adaptive {n}-color palette (k-means)…", end="", flush=True)
    pixels = srgb_to_linear(np.array(img.convert("RGB"), dtype=np.float32))
    flat   = pixels.reshape(-1, 3)
    step   = max(1, len(flat) // 20_000)
    km     = MiniBatchKMeans(n_clusters=n, n_init=3, random_state=42)
    km.fit(flat[::step])
    print(" done")
    return km.cluster_centers_.astype(np.float32)  # linear float32 (n, 3)

def build_fixed_palette(colors: list) -> np.ndarray:
    """Convert list of sRGB [0-255] triples → linear float32 palette."""
    return srgb_to_linear(np.array(colors, dtype=np.float32))


# ══════════════════════════════════════════════════════════════════════════════
# Progress helper
# ══════════════════════════════════════════════════════════════════════════════

def _progress(y: int, h: int, label: str) -> None:
    pct = int(y / h * 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  dithering [{bar}] {pct}% ({label})", end="", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 1 – Floyd-Steinberg
# ══════════════════════════════════════════════════════════════════════════════

def dither_floyd_steinberg(arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """arr: (H,W,3) linear float32. Returns (H,W,3) linear float32."""
    h, w = arr.shape[:2]
    buf  = arr.copy()
    out  = np.empty_like(buf)

    for y in range(h):
        if y % 100 == 0:
            _progress(y, h, "floyd-steinberg")
        for x in range(w):
            old   = buf[y, x].copy()
            dists = ((palette - old) ** 2).sum(axis=1)
            new   = palette[dists.argmin()]
            out[y, x] = new
            err   = old - new
            if x + 1 < w:
                buf[y,   x+1] += err * (7/16)
            if y + 1 < h:
                if x - 1 >= 0:
                    buf[y+1, x-1] += err * (3/16)
                buf[y+1, x  ] += err * (5/16)
                if x + 1 < w:
                    buf[y+1, x+1] += err * (1/16)

    _progress(h, h, "floyd-steinberg")
    print()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 2 – Jarvis-Judice-Ninke  (12-neighbour spread)
# ══════════════════════════════════════════════════════════════════════════════

def dither_jarvis(arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    buf  = arr.copy()
    out  = np.empty_like(buf)

    # (dy, dx, weight/48)
    spread = [
        (0,  1, 7), (0,  2, 5),
        (1, -2, 3), (1, -1, 5), (1, 0, 7), (1, 1, 5), (1, 2, 3),
        (2, -2, 1), (2, -1, 3), (2, 0, 5), (2, 1, 3), (2, 2, 1),
    ]

    for y in range(h):
        if y % 100 == 0:
            _progress(y, h, "jarvis")
        for x in range(w):
            old   = buf[y, x].copy()
            dists = ((palette - old) ** 2).sum(axis=1)
            new   = palette[dists.argmin()]
            out[y, x] = new
            err   = old - new
            for dy, dx, wt in spread:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    buf[ny, nx] += err * (wt / 48)

    _progress(h, h, "jarvis")
    print()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 3 – Ostromoukhov  (variable weights per luminance level)
# ══════════════════════════════════════════════════════════════════════════════

def _ostro_weights(luminance: float) -> tuple[float, float, float]:
    """
    Return (right, down-left, down) error fractions.
    Weights vary continuously with pixel luminance — darker pixels spread
    more downward, brighter pixels spread more right, reducing banding.
    """
    t  = float(np.clip(luminance, 0.0, 1.0))
    r  = 0.4375 + 0.10 * (t - 0.5)
    dl = 0.1875 - 0.05 * (t - 0.5)
    d  = 0.3125 - 0.05 * (t - 0.5)
    total = r + dl + d
    return r / total, dl / total, d / total

def dither_ostromoukhov(arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    buf  = arr.copy()
    out  = np.empty_like(buf)

    for y in range(h):
        if y % 100 == 0:
            _progress(y, h, "ostromoukhov")
        for x in range(w):
            old   = buf[y, x].copy()
            dists = ((palette - old) ** 2).sum(axis=1)
            new   = palette[dists.argmin()]
            out[y, x] = new
            err   = old - new
            lum   = float(np.dot(np.clip(old, 0, 1), [0.2126, 0.7152, 0.0722]))
            wr, wdl, wd = _ostro_weights(lum)
            if x + 1 < w:
                buf[y,   x+1] += err * wr
            if y + 1 < h:
                buf[y+1, x  ] += err * wd
                if x - 1 >= 0:
                    buf[y+1, x-1] += err * wdl

    _progress(h, h, "ostromoukhov")
    print()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Algorithm 4 – Blue-Noise  (void-and-cluster threshold matrix)
# ══════════════════════════════════════════════════════════════════════════════

_BLUE_NOISE_MATRIX: np.ndarray | None = None

def _generate_blue_noise_matrix(size: int = 64) -> np.ndarray:
    """Void-and-cluster blue-noise threshold matrix in [0,1]."""
    from scipy.ndimage import gaussian_filter
    rng    = np.random.default_rng(42)
    N      = size * size
    sigma  = size / 16.0
    binary = np.zeros((size, size), dtype=bool)
    # seed ~10% ones
    ones   = rng.choice(N, N // 10, replace=False)
    binary.flat[ones] = True

    rank   = np.empty(N, dtype=np.int32)
    filled = 0

    # Phase 1: remove ones from tightest cluster, record order
    temp = binary.copy()
    for i in range(N // 10):
        e   = gaussian_filter(temp.astype(np.float32), sigma=sigma, mode="wrap")
        pos = int(np.where(temp, e, -np.inf).argmax())
        rank[filled] = pos
        filled += 1
        temp.flat[pos] = False

    # Phase 2: add zeros into largest void
    temp = binary.copy()
    for i in range(N - N // 10):
        e   = gaussian_filter(temp.astype(np.float32), sigma=sigma, mode="wrap")
        pos = int(np.where(~temp, e, np.inf).argmin())
        rank[filled] = pos
        filled += 1
        temp.flat[pos] = True

    threshold          = np.empty(N, dtype=np.float32)
    threshold[rank[:]] = np.arange(N, dtype=np.float32) / N
    return threshold.reshape(size, size)

def _get_blue_noise(size: int = 64) -> np.ndarray:
    global _BLUE_NOISE_MATRIX
    if _BLUE_NOISE_MATRIX is None:
        print("  🔵 generating blue-noise matrix…", end="", flush=True)
        try:
            _BLUE_NOISE_MATRIX = _generate_blue_noise_matrix(size)
            print(" done")
        except ImportError:
            print(" (scipy missing, using Bayer fallback)")
            bayer = np.array([
                [ 0,32, 8,40, 2,34,10,42],
                [48,16,56,24,50,18,58,26],
                [12,44, 4,36,14,46, 6,38],
                [60,28,52,20,62,30,54,22],
                [ 3,35,11,43, 1,33, 9,41],
                [51,19,59,27,49,17,57,25],
                [15,47, 7,39,13,45, 5,37],
                [63,31,55,23,61,29,53,21],
            ], dtype=np.float32) / 64.0
            r = (size // 8) + 1
            _BLUE_NOISE_MATRIX = np.tile(bayer, (r, r))[:size, :size]
    return _BLUE_NOISE_MATRIX

def dither_blue_noise(arr: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Threshold dithering with a void-and-cluster blue-noise matrix.
    No error propagation → zero worm/streak artifacts.
    Perturb each pixel by noise offset before snapping to nearest palette color.
    """
    h, w   = arr.shape[:2]
    matrix = _get_blue_noise(64)
    mh, mw = matrix.shape

    ty    = math.ceil(h / mh)
    tx    = math.ceil(w / mw)
    tiled = np.tile(matrix, (ty, tx))[:h, :w, None]   # (H, W, 1)

    noise_scale = 0.18
    perturbed   = np.clip(arr + (tiled - 0.5) * noise_scale, 0.0, 1.0)

    # Vectorised nearest-colour
    flat  = perturbed.reshape(-1, 3)
    dists = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    idx   = dists.argmin(axis=1)
    out   = palette[idx].reshape(h, w, 3)

    print(f"\r  dithering [{'█'*20}] 100% (blue-noise)   ")
    return out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

ALGORITHMS = {
    "floyd_steinberg": dither_floyd_steinberg,
    "jarvis":          dither_jarvis,
    "ostromoukhov":    dither_ostromoukhov,
    "blue_noise":      dither_blue_noise,
}

def dither(img: Image.Image, palette_linear: np.ndarray, algorithm: str) -> Image.Image:
    """Full pipeline: sRGB → linear → dither → sRGB."""
    fn = ALGORITHMS.get(algorithm)
    if fn is None:
        print(f"⚠️  Unknown algorithm '{algorithm}', falling back to blue_noise.")
        fn = dither_blue_noise

    arr_linear    = srgb_to_linear(np.array(img.convert("RGB"), dtype=np.float32))
    result_linear = fn(arr_linear, palette_linear)
    return Image.fromarray(linear_to_srgb(result_linear), "RGB")


# ══════════════════════════════════════════════════════════════════════════════
# Crop
# ══════════════════════════════════════════════════════════════════════════════

def smart_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src_w, src_h = img.size
    scale  = max(target_w / src_w, target_h / src_h)
    new_w  = int(src_w * scale)
    new_h  = int(src_h * scale)
    img    = img.resize((new_w, new_h), Image.LANCZOS)
    left   = (new_w - target_w) // 2
    top    = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


# ══════════════════════════════════════════════════════════════════════════════
# Main processing loop
# ══════════════════════════════════════════════════════════════════════════════

def process_folder(cfg: dict, folder: str) -> None:
    src_dir   = Path(folder).resolve()
    target_w  = cfg["width"]
    target_h  = cfg["height"]
    supported = set(cfg["extensions"])
    algorithm = cfg["dither_algorithm"]
    pal_mode  = cfg["palette_mode"]
    n_colors  = cfg["num_colors"]

    if not src_dir.is_dir():
        print(f"❌  Not a directory: {src_dir}")
        sys.exit(1)

    out_dir = src_dir / "img"
    out_dir.mkdir(exist_ok=True)

    print(f"📂  Source    : {src_dir}")
    print(f"📁  Output    : {out_dir}")
    print(f"📐  Size      : {target_w}×{target_h}")
    print(f"🎨  Palette   : {pal_mode} ({n_colors} colors)")
    print(f"🔀  Algorithm : {algorithm}")
    print(f"🔍  Exts      : {', '.join(sorted(supported))}\n")

    images = [p for p in src_dir.iterdir() if p.suffix.lower() in supported]
    if not images:
        print("⚠️  No supported images found.")
        return

    fixed_palette = None
    if pal_mode == "fixed":
        fixed_palette = build_fixed_palette(cfg["palette"])

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}")
        try:
            with Image.open(img_path) as img:
                print(f"  cropping  {img.size} → ({target_w}×{target_h})")
                cropped = smart_crop(img, target_w, target_h)

                palette = (
                    build_adaptive_palette(cropped, n_colors)
                    if pal_mode == "adaptive"
                    else fixed_palette
                )

                result = dither(cropped, palette, algorithm)

                out_name = img_path.stem + "_dithered.png"
                out_path = out_dir / out_name
                result.save(out_path, "PNG", optimize=True)
                kb = out_path.stat().st_size // 1024
                print(f"  ✅ saved  → {out_name}  ({kb} KB)\n")

        except Exception as e:
            import traceback
            print(f"  ⚠️  Skipped: {e}")
            traceback.print_exc()
            print()

    print(f"Done! {len(images)} image(s) processed → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg    = load_config()
    folder = sys.argv[1] if len(sys.argv) > 1 else cfg["photo_dir"]
    process_folder(cfg, folder)


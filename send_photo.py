#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "Pillow",
#     "numpy",
# ]
# ///

"""
Process and send the next photo (circular) from a folder to an e-ink ESP32 display.
Usage: ./send_photo.py [folder_path]

Reads config.json from the same directory as the script.
If folder_path argument is omitted, uses "output_dir" from config.json.

State is kept in <output_dir>/.last_sent — the filename of the last sent photo.
Each run advances to the next file in sorted order, wrapping around.

Pixel format: 4-bit packed, 2 pixels per byte, high nibble first.
Each nibble is a palette index (0–N) matching the palette in config.json.
"""

import json
import sys
import requests
import numpy as np
from pathlib import Path
from PIL import Image

CONFIG_PATH = Path(__file__).parent / "config.json"
CACHE_FILE  = ".last_sent"
DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    print(f"Warning: config.json not found at {CONFIG_PATH}, using defaults.")
    return {}


def list_photos(folder: Path, extensions: set[str], sort_name: bool = True) -> list[Path]:
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in extensions]
    if sort_name:
        files.sort(key=lambda f: f.name)
    return files


def pick_next_photo(folder: Path, photos: list[Path]) -> Path:
    """Read cache, find next photo in circular order, save new cache."""
    cache_path = folder / CACHE_FILE
    last_name = None

    if cache_path.exists():
        last_name = cache_path.read_text().strip()

    names = [p.name for p in photos]

    if last_name and last_name in names:
        idx = (names.index(last_name) + 1) % len(photos)
    else:
        idx = 0  # first run or file was removed

    chosen = photos[idx]
    cache_path.write_text(chosen.name)
    print(f"  Cache: {last_name or '(none)'} → {chosen.name}  [{idx + 1}/{len(photos)}]")
    return chosen


def dither_to_palette(img: Image.Image, palette_rgb: list[list[int]]) -> np.ndarray:
    """Floyd-Steinberg dither to fixed palette. Returns index array (H, W)."""
    arr = np.array(img, dtype=np.float32)
    pal = np.array(palette_rgb, dtype=np.float32)
    h, w, _ = arr.shape
    indices = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            old = arr[y, x].copy()
            dists = np.sum((pal - old) ** 2, axis=1)
            idx = int(np.argmin(dists))
            indices[y, x] = idx
            err = old - pal[idx]
            arr[y, x] = pal[idx]

            if x + 1 < w:
                arr[y, x + 1]     += err * 7 / 16
            if y + 1 < h:
                if x > 0:
                    arr[y + 1, x - 1] += err * 3 / 16
                arr[y + 1, x]     += err * 5 / 16
                if x + 1 < w:
                    arr[y + 1, x + 1] += err * 1 / 16

    return indices


def pack_4bit(indices: np.ndarray) -> bytes:
    """Pack index array into 4-bit nibbles, high nibble = left pixel."""
    flat = indices.flatten()
    if len(flat) % 2 != 0:
        flat = np.append(flat, 0)
    packed = ((flat[0::2].astype(np.uint8) << 4) | (flat[1::2].astype(np.uint8) & 0x0F))
    return packed.tobytes()


def process_image(image_path: Path, config: dict) -> bytes:
    width       = config.get("width", 1200)
    height      = config.get("height", 1600)
    palette_rgb = config.get("palette", [
        [0, 0, 0], [255, 255, 255], [255, 0, 0],
        [0, 255, 0], [0, 0, 255], [255, 255, 0],
    ])

    img = Image.open(image_path).convert("RGB")
    print(f"  Original size : {img.size}")

    img.thumbnail((width, height), Image.LANCZOS)
    if img.size != (width, height):
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        offset = ((width - img.width) // 2, (height - img.height) // 2)
        canvas.paste(img, offset)
        img = canvas

    print(f"  Resized to    : {img.size}")
    print(f"  Dithering to {len(palette_rgb)} palette colors ...")

    indices = dither_to_palette(img, palette_rgb)
    raw = pack_4bit(indices)

    expected = (width * height + 1) // 2
    print(f"  Packed bytes  : {len(raw)} (expected {expected})")
    return raw


def send_photo(image_path: Path, config: dict, host: str) -> None:
    print(f"Processing: {image_path.name}")
    raw_data = process_image(image_path, config)

    server_url = f"{host}/upload"
    print(f"Sending {len(raw_data)} bytes to {server_url} ...")
    files = {
        "data": ("image_data.bin", raw_data, "application/octet-stream")
    }
    headers = {
        "Accept": "*/*",
        "Cache-Control": "no-cache",
        "Origin": host,
        "Referer": f"{host}/wifi",
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        ),
    }

    try:
        response = requests.post(
            server_url, files=files, headers=headers, verify=False, timeout=60,
        )
    except requests.exceptions.ConnectionError as e:
        print(f"Error: Cannot connect to {SERVER_URL}")
        print("  Make sure the device is powered on and on the same network.")
        raise SystemExit(1) from e

    ok = response.ok and "fail" not in response.text.lower()
    if ok:
        print(f"Success: {response.status_code} — {response.text.strip() or 'photo uploaded'}")
    else:
        print(f"Failed : status {response.status_code} — {response.text[:500]}")
        if not response.ok:
            response.raise_for_status()
        raise SystemExit(1)


def main() -> None:
    config = load_config()

    host = config.get("host").rstrip("/")

    if len(sys.argv) > 1:
        folder = Path(sys.argv[1]).resolve()
    elif "output_dir" in config:
        folder = Path(config["output_dir"]).resolve()
        print(f"Using output_dir from config.json: {folder}")
    else:
        folder = Path(".")

    if not folder.is_dir():
        print(f"Error: '{folder}' is not a directory.")
        sys.exit(1)

    extensions = set(config.get("extensions", DEFAULT_EXTENSIONS))
    sort_name  = config.get("sort_name", True)

    photos = list_photos(folder, extensions, sort_name)
    if not photos:
        print(f"No image files found in '{folder}' with extensions {extensions}.")
        sys.exit(1)

    photo = pick_next_photo(folder, photos)
    print(f"Sending: {photo.name} from {photo.parent}")
    send_photo(photo, config, host)
    print("Done ✓")


if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()
# 🖼️ process_images

Batch image processor that crops photos to a target size and reduces them to N colors using pre-dithering. Built for e-ink displays, pixel art, and low-color output targets.

---

## Features

- Smart center-crop to any target resolution
- 4 dithering algorithms, selectable per run
- Adaptive palette via k-means (picks best colors per image) or fixed palette
- Proper gamma correction — dithering runs in linear light space, not sRGB
- Zero config required — sensible defaults built in
- Outputs saved to `img/` subfolder inside source directory

---

## Requirements

- [uv](https://github.com/astral-sh/uv) — dependencies install automatically on first run

---

## Usage

```bash
# Run with config.json or built-in defaults
./process_images.py

# Override the source folder via CLI
./process_images.py /path/to/your/photos

# Or explicitly via uv
uv run process_images.py /path/to/your/photos
```

Output images are saved to `<photo_dir>/img/` as `<original_name>_dithered.png`.

---

## Configuration

Place a `config.json` next to the script. All keys are optional — missing keys fall back to defaults.

```json
{
  "photo_dir":        "./photos",
  "extensions":       [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"],
  "width":            1200,
  "height":           1600,
  "num_colors":       6,
  "dither_algorithm": "blue_noise",
  "palette_mode":     "adaptive",
  "palette": [
    [0,   0,   0  ],
    [255, 255, 255],
    [255, 0,   0  ],
    [0,   255, 0  ],
    [0,   0,   255],
    [255, 255, 0  ]
  ]
}
```

### Config reference

| Key | Default | Description |
|---|---|---|
| `photo_dir` | `./photos` | Folder containing source images |
| `output_dir` | `""` | Output folder path; empty = `<photo_dir>/img/` |
| `extensions` | common formats | File extensions to process |
| `width` | `1200` | Output width in pixels |
| `height` | `1600` | Output height in pixels |
| `num_colors` | `6` | Number of colors in output |
| `dither_algorithm` | `blue_noise` | See algorithms below |
| `palette_mode` | `adaptive` | `adaptive` or `fixed` |
| `palette` | 6 basic colors | Used only when `palette_mode` is `fixed` |
| `sort_name` | `false` | If `true`, rename outputs to `01.png`, `02.png`, … |
| `png_compression` | `6` | PNG compression level 0–9 (0 = fastest/largest, 9 = slowest/smallest) |
| `png_optimize` | `true` | Extra compression pass — smaller files, slightly slower |

---

## Dithering Algorithms

### `blue_noise` ⭐ (default)
Threshold dithering using a void-and-cluster blue-noise matrix. No error propagation means zero worm or streak artifacts. Most natural-looking result, especially with few colors.

### `ostromoukhov`
Error diffusion with variable weights that change based on pixel luminance — darker pixels spread error more downward, brighter pixels more rightward. Near-optimal for classic diffusion. Reduces the banding common in fixed-weight algorithms.

### `jarvis`
Jarvis-Judice-Ninke algorithm. Spreads error to 12 neighboring pixels (vs 4 in Floyd-Steinberg), producing smoother gradients. Good for images with large smooth areas like skies or skin.

### `floyd_steinberg`
The classic. Fast and well-understood. With very few colors (≤6) can produce visible diagonal "worm" artifacts in smooth gradients. Best used when speed matters over quality.

---

## Palette Modes

### `adaptive` ⭐ (default)
Runs k-means clustering on each image in linear RGB space to find the N colors that best represent that specific image. A forest photo gets greens and browns; a sunset gets oranges and purples. Produces the highest quality output.

### `fixed`
Uses the color list in the `palette` config key. Useful for hardware targets with a fixed set of colors, such as 7-color e-ink displays.

---

## Gamma Correction

All dithering runs in **linear light space**, not sRGB. This means:

- Color distances are perceptually accurate
- Dark areas stay detailed instead of going muddy
- Gradients dither smoothly across the full luminance range

Conversion back to sRGB happens after dithering.

---

## Algorithm Comparison

| Algorithm | Quality | Speed | Best for |
|---|---|---|---|
| `blue_noise` | ⭐⭐⭐⭐⭐ | ⚡⚡⚡⚡ | Any image — default choice |
| `ostromoukhov` | ⭐⭐⭐⭐ | ⚡⚡ | High detail, portraits |
| `jarvis` | ⭐⭐⭐ | ⚡⚡ | Smooth gradients, landscapes |
| `floyd_steinberg` | ⭐⭐ | ⚡⚡⚡⚡⚡ | Speed priority, many colors |

**Biggest quality win:** switch `palette_mode` from `fixed` to `adaptive`. Letting k-means pick the best colors per image makes a larger difference than any algorithm choice.

**Second biggest win:** `blue_noise` over `floyd_steinberg`. Error diffusion algorithms propagate mistakes pixel-to-pixel, which creates directional streaks. Blue-noise threshold dithering has no error propagation at all — noise is spatially uncorrelated by construction, so patterns never form.

**For e-ink displays** with a hardware-fixed palette (e.g. Waveshare 7-color): use `palette_mode: fixed` with your display's exact RGB values, and `blue_noise` or `ostromoukhov` for the algorithm.

---

## Example Output

```
ℹ️  No config.json found, using built-in defaults.
📂  Source    : /Users/you/photos
📁  Output    : /Users/you/photos/img
📐  Size      : 1200×1600
🎨  Palette   : adaptive (6 colors)
🔀  Algorithm : blue_noise
🔍  Exts      : .bmp, .jpeg, .jpg, .png, .tif, .tiff, .webp

[1/3] sunset.jpg
  cropping  (4032×3024) → (1200×1600)
  🎨 building adaptive 6-color palette (k-means)… done
  🔵 generating blue-noise matrix… done
  dithering [████████████████████] 100% (blue-noise)
  ✅ saved  → sunset_dithered.png  (142 KB)
```
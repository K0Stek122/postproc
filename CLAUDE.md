# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`postproc` is a document/photo post-processing CLI tool that scans, crops, denoises, and enhances images (e.g. photographed documents). It supports single-file and batch directory processing via multiprocessing.

## Running

```bash
# Single image
python postproc.py -i input.jpg -o output.jpg

# With a config file
python postproc.py -i input.jpg -o output.jpg -c cfg.json

# Batch directory (parallel processing)
python postproc.py -i ./input_dir -o ./output_dir -c cfg.json
```

## Dependencies

```bash
pip install -r requirements.txt
```

Only `opencv-python` is required; `numpy` is a transitive dependency.

## Architecture

All logic lives in `postproc.py`. Entry point is at module level (lines 107–132) — argument parsing, optional config loading, then dispatch to single-file or batch mode.

**`Config` dataclass** — holds all tunable parameters. Loaded from a JSON file via `Config(**json.load(f))` or uses defaults.

**`Scanner` class** — the core processing pipeline, run in `__init__`:
1. `_preprocess` — grayscale → Gaussian blur → Otsu threshold
2. `_define_contour` — finds external contours, filters noise (<1% of image area)
3. `_create_coordinates_from_contour` — merges all contours into a single bounding box
4. `_crop_image` — crops original (color) image to that bounding box
5. `_denoise` — `fastNlMeansDenoisingColored`
6. `_weight_image` — blends original color with an adaptive threshold layer (controls how "scanned" it looks)
7. `_rotate_image` — snaps `rotation_degrees` to nearest 90° and rotates

**Batch mode** uses `multiprocessing.Pool(cpu_count())` to process images in parallel.

## Config parameters

| Key | Default | Effect |
|-----|---------|--------|
| `gaussian_strength` | `[5, 5]` | Kernel size for blur before contour detection |
| `denoise_strength` | `10` | `h`/`hColor` for NlMeans denoising |
| `adaptive_threshold_block_size` | `51` | Block size for adaptive threshold (must be odd) |
| `adaptive_threshold_sensitivity` | `15` | Constant subtracted in adaptive threshold |
| `original_color_percentage` | `0.4` | Weight of original color vs. thresholded layer (0=fully thresholded, 1=original) |
| `rotation_degrees` | `270` | Final rotation; snapped to nearest 90° |

Note: `cfg.json` in the repo uses `original_color_percentage: 0.8`, while the `Config` dataclass default is `0.4`.

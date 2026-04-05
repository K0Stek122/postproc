# postproc

A CLI tool for post-processing photos of documents. It automatically detects and crops the document, denoises the image, and blends the original color with a thresholded layer to produce a clean, scan-like result.

## Example

| Before | After |
|--------|-------|
| Raw photo of a document | Cropped, denoised, enhanced |

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and OpenCV.

## Usage

**Single image:**
```bash
python postproc.py -i photo.jpg -o result.jpg
```

**Batch directory** (processed in parallel):
```bash
python postproc.py -i ./photos/ -o ./results/
```

**With a custom config:**
```bash
python postproc.py -i ./photos/ -o ./results/ -c cfg.json
```

Supported input formats: `.jpg`, `.jpeg`, `.png`

## Configuration

All parameters are optional. Pass a JSON file with any subset of the keys below:

```json
{
    "gaussian_strength": [5, 5],
    "denoise_strength": 10,
    "adaptive_threshold_block_size": 51,
    "adaptive_threshold_sensitivity": 15,
    "original_color_percentage": 0.4,
    "rotation_degrees": 270
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gaussian_strength` | `[5, 5]` | Gaussian blur kernel size applied before contour detection |
| `denoise_strength` | `10` | Denoising intensity (`h`/`hColor` for Non-Local Means) |
| `adaptive_threshold_block_size` | `51` | Block size for adaptive thresholding (must be odd) |
| `adaptive_threshold_sensitivity` | `15` | Constant subtracted during adaptive thresholding |
| `original_color_percentage` | `0.4` | Blend ratio between original color (`1.0`) and fully thresholded (`0.0`) |
| `rotation_degrees` | `270` | Rotation applied to the final image, snapped to the nearest 90° |

The `original_color_percentage` is the most impactful parameter for the final look: lower values produce a high-contrast black-and-white scan appearance, higher values preserve more of the original photo color.

## How it works

Each image goes through the following pipeline:

1. **Contour detection** — converts to grayscale, applies Gaussian blur, then Otsu thresholding to find the document edges
2. **Crop** — all detected contours (filtered to >1% of image area) are merged into a single bounding box and the image is cropped to it
3. **Denoise** — Non-Local Means denoising is applied to the cropped color image
4. **Enhance** — an adaptive threshold layer is blended with the denoised image, controlled by `original_color_percentage`
5. **Rotate** — the image is rotated by `rotation_degrees`

Batch mode distributes images across all available CPU cores using `multiprocessing.Pool`.

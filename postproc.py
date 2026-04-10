import cv2
import numpy as np
import argparse
import json
import os
from multiprocessing import Pool, cpu_count

DEFAULT_CONFIG = {
    "CropStage":    {"gaussian_strength": [5, 5]},
    "DenoiseStage": {"denoise_strength": 10},
    "ClaheStage":   {"clahe_clip_limit": 2.0, "clahe_tile_grid_size": [8, 8]},
    "WeightStage":  {"adaptive_threshold_block_size": 51, "adaptive_threshold_sensitivity": 15, "original_color_percentage": 0.4},
    "RotateStage":  {"rotation_degrees": 270},
    "PerspectiveCropStage" : {"gaussian_strength": [5, 5]},
}


def setup_arguments():
    parser = argparse.ArgumentParser(description='Document post-processing tool')
    parser.add_argument('-i', '--input', required=True, help='Input file or directory')
    parser.add_argument('-c', '--config', help='Path to JSON config file')
    parser.add_argument('-o', '--output', required=True, help="Output file or directory.")
    return parser.parse_args()


class Stage:
    def __init__(self, params: dict):
        self.params = params

    def process(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CropStage(Stage):
    def process(self, img: np.ndarray) -> np.ndarray:
        gaussian_strength = tuple(self.params.get("gaussian_strength", [5, 5]))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, gaussian_strength, 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = img.shape[0] * img.shape[1] * 0.01
        significant = [c for c in contours if cv2.contourArea(c) > min_area]

        merged = np.vstack(significant)
        x, y, w, h = cv2.boundingRect(merged)
        return img[y:y+h, x:x+w]


class PerspectiveCropStage(Stage):
    def process(self, img: np.ndarray) -> np.ndarray:
        gaussian_strength = tuple(self.params.get("gaussian_strength", [5, 5]))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, gaussian_strength, 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = img.shape[0] * img.shape[1] * 0.01
        significant = [c for c in contours if cv2.contourArea(c) > min_area]

        # Use the largest contour and approximate to a polygon.
        # Increase epsilon until we get exactly 4 corners (or give up and use the bounding box).
        largest = max(significant, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        while len(approx) > 4:
            epsilon *= 1.1
            approx = cv2.approxPolyDP(largest, epsilon, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
        else:
            # Fewer than 4 points — fall back to the axis-aligned bounding box
            x, y, w, h = cv2.boundingRect(largest)
            corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)

        # Order corners: top-left, top-right, bottom-right, bottom-left
        s = corners.sum(axis=1)
        d = np.diff(corners, axis=1)
        ordered = np.array([
            corners[np.argmin(s)],   # top-left     (smallest x+y)
            corners[np.argmin(d)],   # top-right     (smallest x-y)
            corners[np.argmax(s)],   # bottom-right  (largest x+y)
            corners[np.argmax(d)],   # bottom-left   (largest x-y)
        ], dtype=np.float32)

        tl, tr, br, bl = ordered
        width  = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(img, M, (width, height))


class DenoiseStage(Stage):
    def process(self, img: np.ndarray) -> np.ndarray:
        strength = self.params.get("denoise_strength", 10)
        return cv2.fastNlMeansDenoisingColored(img, h=strength, hColor=strength)

class ClaheStage(Stage):
    def process(self, img: np.ndarray) -> np.ndarray:
        clip_limit = self.params.get("clahe_clip_limit", 2.0)
        tile_grid_size = tuple(self.params.get("clahe_tile_grid_size", [8, 8]))

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


class WeightStage(Stage):
    def process(self, img: np.ndarray) -> np.ndarray:
        block_size = self.params.get("adaptive_threshold_block_size", 51)
        sensitivity = self.params.get("adaptive_threshold_sensitivity", 15)
        color_pct = self.params.get("original_color_percentage", 0.4)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, sensitivity)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img, color_pct, thresh_bgr, 1 - color_pct, 0)


class RotateStage(Stage):
    def process(self, img: np.ndarray) -> np.ndarray:
        degrees = self.params.get("rotation_degrees", 270)
        _rotation_map = {
            90:  cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        snapped = (round(degrees / 90) * 90) % 360
        if snapped in _rotation_map:
            return cv2.rotate(img, _rotation_map[snapped])
        return img


class Scanner:
    _STAGE_REGISTRY = {
        "CropStage":    CropStage,
        "DenoiseStage": DenoiseStage,
        "ClaheStage":   ClaheStage,
        "WeightStage":  WeightStage,
        "RotateStage":  RotateStage,
        "PerspectiveCropStage" : PerspectiveCropStage
    }

    def __init__(self, img_dir, pipeline: dict = None):
        self.pipeline = pipeline or DEFAULT_CONFIG
        self.img = cv2.imread(img_dir)
        self.final_img = self._process()

    def _process(self):
        img = self.img
        for stage_name, params in self.pipeline.items():
            img = self._STAGE_REGISTRY[stage_name](params).process(img)
        return img

    def display_photo(self):
        """Display the final processed image."""
        cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('pic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('pic', self.final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, output_path: str):
        """Save the final processed image to disk."""
        cv2.imwrite(output_path, self.final_img)


args = setup_arguments()

pipeline = None
if args.config:
    with open(args.config) as f:
        pipeline = json.load(f)

def process_image(input_path, output_path, pipeline):
    print(f"Processing {os.path.basename(input_path)}...")
    Scanner(input_path, pipeline).save_image(output_path)

if os.path.isdir(args.input):
    os.makedirs(args.output, exist_ok=True)
    images = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    tasks = [(os.path.join(args.input, f), os.path.join(args.output, f), pipeline) for f in images]
    with Pool(cpu_count()) as pool:
        pool.starmap(process_image, tasks)
    print(f"Done. {len(images)} image(s) saved to {args.output}")
else:
    Scanner(args.input, pipeline).save_image(args.output)
    print("Done.")

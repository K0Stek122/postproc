import cv2
import numpy as np
import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass

@dataclass
class Config:
    """Class for tracking config values"""
    gaussian_strength: tuple[int, int] = (5, 5)
    denoise_strength: int = 10
    adaptive_threshold_block_size: int = 51
    adaptive_threshold_sensitivity: int = 15
    original_color_percentage: float = 0.4
    rotation_degrees: int = 270


def setup_arguments():
    parser = argparse.ArgumentParser(description='Document post-processing tool')
    parser.add_argument('-i', '--input', required=True, help='Input file or directory')
    parser.add_argument('-c', '--config', help='Path to JSON config file')
    parser.add_argument('-o', '--output', required=True, help="Output file or directory.")
    parser.add_argument('-m', '--mode', choices=['full', 'crop-only'], default='full', help="Processing mode: 'full' (default) or 'crop-only'.")
    return parser.parse_args()

class Scanner:
    def __init__(self, img_dir, cfg: Config = None, mode: str = 'full'):
        self.cfg = cfg or Config()
        self.mode = mode
        self.img = cv2.imread(img_dir)
        self.final_img = self._process()

    def _process(self):
        thresh = self._preprocess()
        contour = self._define_contour(thresh)
        coords = self._create_coordinates_from_contour(contour)
        cropped = self._crop_image(coords)
        rotated = self._rotate_image(cropped)
        if self.mode == 'crop-only':
            return rotated
        denoised = self._denoise(rotated)
        weighted = self._weight_image(denoised)
        return weighted

    def display_photo(self):
        """Display the final processed image."""
        cv2.namedWindow('pic', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('pic', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('pic', self.final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _preprocess(self):
        gray = self._to_grayscale()
        blurred = self._apply_gaussian_blur(gray)
        return self._apply_threshold(blurred)

    def _to_grayscale(self):
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def _apply_gaussian_blur(self, img):
        return cv2.GaussianBlur(img, self.cfg.gaussian_strength, 0)

    def _apply_threshold(self, img):
        # Threshold classifies each pixel as black or white based on local intensity
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _define_contour(self, thresh):
        """Returns all significant contours, filtering out noise below 1% of image area."""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.img.shape[0] * self.img.shape[1] * 0.01
        return [c for c in contours if cv2.contourArea(c) > min_area]

    def _create_coordinates_from_contour(self, contours):
        """Returns a bounding box that encompasses all given contours."""
        merged = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(merged)
        return (x, y, w, h)

    def _crop_image(self, coords):
        x, y, w, h = coords
        return self.img[y:y+h, x:x+w]

    def _denoise(self, img):
        return cv2.fastNlMeansDenoisingColored(img, h=self.cfg.denoise_strength, hColor=self.cfg.denoise_strength)

    def _weight_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.cfg.adaptive_threshold_block_size, self.cfg.adaptive_threshold_sensitivity)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img, self.cfg.original_color_percentage, thresh_bgr, 1 - self.cfg.original_color_percentage, 0)

    def _rotate_image(self, img):
        """Rotate the final image by cfg.rotation_degrees, snapped to the nearest 90°."""
        _rotation_map = {
            90:  cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        snapped = (round(self.cfg.rotation_degrees / 90) * 90) % 360
        if snapped in _rotation_map:
            return cv2.rotate(img, _rotation_map[snapped])
        return img

    def save_image(self, output_path: str):
        """Save the final processed image to disk."""
        cv2.imwrite(output_path, self.final_img)

args = setup_arguments()

cfg = Config()
if args.config:
    with open(args.config) as f:
        cfg = Config(**json.load(f))

def process_image(input_path, output_path, cfg, mode):
    print(f"Processing {os.path.basename(input_path)}...")
    Scanner(input_path, cfg, mode).save_image(output_path)

if os.path.isdir(args.input):
    # Create the directory for output
    os.makedirs(args.output, exist_ok=True)

    # grab all of them images from input directory
    images = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Create a task for each image for later Pooling
    tasks = [(os.path.join(args.input, f), os.path.join(args.output, f), cfg, args.mode) for f in images]
    with Pool(cpu_count()) as pool:
        pool.starmap(process_image, tasks)
    print(f"Done. {len(images)} image(s) saved to {args.output}")
else:
    Scanner(args.input, cfg, args.mode).save_image(args.output)
    print("Done.")

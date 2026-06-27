"""
lib/manhwa_panel_detector.py — OpenCV-based panel detection for manhwa/manga.

Used in `--mode manhwa_panels` pipeline.
Instead of using an LLM to summarize text and a diffusion model to generate new images,
this mode simply detects the original drawn panels, crops them, and feeds them
directly to the MotionEngine (Ken Burns, panning) to create a video.

Much faster, much cheaper, and perfectly preserves the original art.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from lib.config import PANEL_DETECTION_THRESHOLD

logger = logging.getLogger(__name__)


class ManhwaPanelDetector:
    """
    Detects and crops individual panels from a full manhwa/manga page.
    """

    def __init__(self, project_name: str, projects_root: str = "projects"):
        self.project_name = project_name
        self.project_dir = Path(projects_root) / project_name
        self.panels_dir = self.project_dir / "images"  # Output directory (same as generated images)
        self.panels_dir.mkdir(parents=True, exist_ok=True)

    def detect_panels(self, image_path: Path) -> List[np.ndarray]:
        """
        Detect panels in an image using contour detection.
        Returns a list of cropped panel images (as numpy arrays, BGR format).
        Sorted top-to-bottom.
        """
        try:
            # Read image
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Failed to read image: {image_path}")
                return []

            # Preprocessing for contour detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Thresholding (assuming white background with black panel borders)
            # Inverse binary threshold: borders become white, background black
            _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Morphological operations to close gaps in panel borders
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            
            # Find contours
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours based on area
            h, w = img.shape[:2]
            img_area = h * w
            
            # Convert PANEL_DETECTION_THRESHOLD (e.g. 50 meaning 1/50th of image) to an absolute minimum area
            min_area = img_area / PANEL_DETECTION_THRESHOLD
            
            panels_rects = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    x, y, pw, ph = cv2.boundingRect(cnt)
                    # Exclude contours that cover almost the entire image (the page border itself)
                    if pw * ph > img_area * 0.95:
                        continue
                    panels_rects.append((x, y, pw, ph))
            
            # If no panels found (maybe no borders, e.g. webtoon style),
            # split the image vertically into chunks as a fallback
            if not panels_rects:
                logger.debug(f"No explicit panels found in {image_path.name}, falling back to chunks.")
                chunk_height = min(h, w) # rough square chunks
                for y in range(0, h, chunk_height):
                    y_end = min(y + chunk_height, h)
                    panels_rects.append((0, y, w, y_end - y))

            # Sort top-to-bottom
            panels_rects.sort(key=lambda r: r[1])

            # Crop panels
            cropped_panels = []
            for (x, y, pw, ph) in panels_rects:
                # Add a small padding (margin) around the detected panel if possible
                margin = 10
                y1 = max(0, y - margin)
                y2 = min(h, y + ph + margin)
                x1 = max(0, x - margin)
                x2 = min(w, x + pw + margin)
                
                cropped = img[y1:y2, x1:x2]
                cropped_panels.append(cropped)

            return cropped_panels

        except Exception as e:
            logger.error(f"Panel detection failed on {image_path}: {e}")
            return []

    def process_chapter(self, image_paths: List[Path], start_index: int = 1) -> List[Tuple[str, Path]]:
        """
        Process a list of chapter pages, extract panels, and save them to disk.
        Returns a list of (segment_id, saved_path) tuples.
        """
        all_panels = []
        global_panel_idx = start_index

        for page_idx, img_path in enumerate(image_paths, 1):
            logger.info(f"Detecting panels in {img_path.name}...")
            panels = self.detect_panels(img_path)
            
            for i, panel in enumerate(panels, 1):
                # Save the panel image
                segment_id = f"{global_panel_idx:04d}"
                out_name = f"scene_{segment_id}.png"
                out_path = self.panels_dir / out_name
                
                # cv2.imwrite expects BGR, which is what we have
                cv2.imwrite(str(out_path), panel)
                all_panels.append((segment_id, out_path))
                global_panel_idx += 1
                
        logger.info(f"Extracted {len(all_panels)} panels in total.")
        return all_panels

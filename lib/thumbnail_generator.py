"""
lib/thumbnail_generator.py — YouTube thumbnail generator.

Generates output/thumbnail.png using:
  1. The key scene image (from PlannerAgent's key_scenes list)
  2. Cinematic color grading (contrast, saturation, vignette)
  3. Title text overlay with shadow

Pure PIL — no model loading. Runs on CPU.
Output: 1280×720 (YouTube standard)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

THUMBNAIL_W = 1280
THUMBNAIL_H = 720
DEFAULT_FONT_SIZE = 72
DEFAULT_TITLE_COLOR = "#FFFFFF"
DEFAULT_SHADOW_COLOR = "#000000"


class ThumbnailGenerator:
    """
    YouTube thumbnail generator.

    Selects the best scene image, applies cinematic grading,
    and overlays the project title.

    Usage
    -----
    gen = ThumbnailGenerator("my_project", "projects")
    path = gen.generate(episode=1, title="My Novel — Episode 1")
    """

    def __init__(
        self,
        project_name: str,
        projects_root: Optional[str] = None,
        font_size: int = DEFAULT_FONT_SIZE,
        title_color: str = DEFAULT_TITLE_COLOR,
        width: int = THUMBNAIL_W,
        height: int = THUMBNAIL_H,
    ):
        root = Path(projects_root) if projects_root else Path("projects")
        self.project_dir = root / project_name
        self.font_size = font_size
        self.title_color = title_color
        self.width = width
        self.height = height

    def generate(
        self,
        episode: int,
        title: str,
        key_scenes: Optional[List[str]] = None,
        director_profile: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """
        Generate thumbnail.png.

        Parameters
        ----------
        episode : int
        title : str
            Title text to overlay
        key_scenes : list, optional
            Preferred scene IDs (from PlannerAgent)
        director_profile : dict, optional
            Used for color palette adjustments

        Returns
        -------
        Path | None
            Path to generated thumbnail, or None if generation failed
        """
        try:
            from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
        except ImportError:
            logger.warning("[ThumbnailGenerator] Pillow not installed — skipping thumbnail")
            return None

        # 1. Find source image
        source_image = self._find_best_image(episode, key_scenes)
        if source_image is None:
            logger.warning("[ThumbnailGenerator] No source image found — skipping thumbnail")
            return None

        # 2. Load and resize
        try:
            img = Image.open(str(source_image)).convert("RGB")
            img = self._crop_to_aspect(img, self.width, self.height)
            img = img.resize((self.width, self.height), Image.LANCZOS)
        except Exception as e:
            logger.error(f"[ThumbnailGenerator] Image load failed: {e}")
            return None

        # 3. Apply cinematic color grading
        profile = director_profile or {}
        color_palette = profile.get("color_palette", "desaturated")
        img = self._apply_grading(img, color_palette)

        # 4. Apply vignette
        img = self._apply_vignette(img)

        # 5. Overlay title text
        img = self._overlay_title(img, title)

        # 6. Save
        output_dir = self.project_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "thumbnail.png"
        img.save(str(out_path), "PNG", optimize=True)

        logger.info(f"[ThumbnailGenerator] Thumbnail saved: {out_path}")
        return out_path

    def _find_best_image(
        self,
        episode: int,
        key_scenes: Optional[List[str]],
    ) -> Optional[Path]:
        """Find the best source image for the thumbnail."""
        images_dir = self.project_dir / "images"
        if not images_dir.exists():
            return None

        # Try key scenes first
        if key_scenes:
            for scene_id in key_scenes:
                for ext in [".png", ".jpg", ".webp"]:
                    candidate = images_dir / f"{scene_id}{ext}"
                    if candidate.exists() and candidate.stat().st_size > 5000:
                        logger.debug(f"[ThumbnailGenerator] Using key scene: {candidate.name}")
                        return candidate

        # Fallback: use the most visually interesting image (by file size as proxy)
        all_images = sorted(
            [p for p in images_dir.glob("*.png") if p.stat().st_size > 5000],
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if all_images:
            logger.debug(f"[ThumbnailGenerator] Using largest image: {all_images[0].name}")
            return all_images[0]

        return None

    def _crop_to_aspect(self, img: Any, target_w: int, target_h: int) -> Any:
        """Center-crop image to target aspect ratio."""
        from PIL import Image
        src_w, src_h = img.size
        target_ratio = target_w / target_h
        src_ratio = src_w / src_h

        if src_ratio > target_ratio:
            # Wider than target — crop width
            new_w = int(src_h * target_ratio)
            offset = (src_w - new_w) // 2
            img = img.crop((offset, 0, offset + new_w, src_h))
        elif src_ratio < target_ratio:
            # Taller than target — crop height
            new_h = int(src_w / target_ratio)
            offset = (src_h - new_h) // 2
            img = img.crop((0, offset, src_w, offset + new_h))

        return img

    def _apply_grading(self, img: Any, color_palette: str) -> Any:
        """Apply cinematic color grading based on style palette."""
        from PIL import ImageEnhance

        # Contrast boost
        contrast_factor = {
            "vibrant": 1.3,
            "bold": 1.4,
            "desaturated": 1.2,
            "muted": 1.1,
            "monochrome": 1.3,
        }.get(color_palette, 1.2)

        img = ImageEnhance.Contrast(img).enhance(contrast_factor)

        # Saturation adjustment
        sat_factor = {
            "vibrant": 1.4,
            "bold": 1.5,
            "desaturated": 0.85,
            "muted": 0.75,
            "monochrome": 0.0,
        }.get(color_palette, 1.1)

        img = ImageEnhance.Color(img).enhance(sat_factor)

        # Brightness (slight)
        img = ImageEnhance.Brightness(img).enhance(1.05)

        return img

    def _apply_vignette(self, img: Any) -> Any:
        """Apply a subtle dark vignette effect around the edges."""
        try:
            from PIL import Image
            import numpy as np

            arr = np.array(img, dtype=float)
            h, w = arr.shape[:2]

            # Create radial gradient mask (1 at center, 0 at edges)
            Y, X = np.ogrid[:h, :w]
            cx, cy = w / 2, h / 2
            dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
            mask = np.clip(1.0 - dist * 0.6, 0.6, 1.0)  # subtle vignette
            mask = mask[:, :, np.newaxis]

            arr = arr * mask
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        except ImportError:
            return img  # numpy not available, skip vignette

    def _overlay_title(self, img: Any, title: str) -> Any:
        """Overlay title text with drop shadow."""
        try:
            from PIL import ImageDraw, ImageFont
        except ImportError:
            return img

        draw = ImageDraw.Draw(img)

        # Try to load a font; fall back to default
        font = None
        try:
            # Try common system font locations
            font_paths = [
                "arial.ttf", "Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ]
            for fp in font_paths:
                try:
                    font = ImageFont.truetype(fp, self.font_size)
                    break
                except Exception:
                    continue
        except Exception:
            pass

        if font is None:
            font = ImageFont.load_default()

        # Measure text
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Position: bottom-center with padding
        x = (self.width - text_w) // 2
        y = self.height - text_h - 60

        # Shadow
        shadow_offset = 3
        draw.text(
            (x + shadow_offset, y + shadow_offset),
            title, font=font,
            fill=DEFAULT_SHADOW_COLOR
        )

        # Title text
        r, g, b = self._hex_to_rgb(self.title_color)
        draw.text((x, y), title, font=font, fill=(r, g, b))

        return img

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert #RRGGBB to (R, G, B)."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

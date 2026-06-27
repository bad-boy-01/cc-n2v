"""
lib/motion_engine.py — Ken Burns + pan effects for storyboard images.

Converts static images into motion video clips using PIL + OpenCV + MoviePy.
No AI video model required. Runs entirely on CPU.

Dynamic duration: each clip duration = max(narration_duration, MIN_SCENE_DURATION)
This synchronizes visuals to audio automatically.

Supported effects (selected from storyboard.json camera_motion field):
  Static, Pan Left, Pan Right, Tilt Up, Tilt Down,
  Zoom In, Zoom Out, Tracking Shot (diagonal pan)

Output: cache/motion/scene_{id}.mp4
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MIN_SCENE_DURATION = 3.0   # seconds — never shorter than this
DEFAULT_FPS = 24
MOTION_ZOOM_FACTOR = 1.12  # how much to zoom for Ken Burns (12%)


def _load_image_as_array(path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    """Load image, resize to target_size (W, H), return uint8 numpy array."""
    img = Image.open(str(path)).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return np.array(img)


def _ken_burns_frames(
    base_arr: np.ndarray,
    n_frames: int,
    direction: str = "zoom_in",
    zoom_factor: float = MOTION_ZOOM_FACTOR,
) -> List[np.ndarray]:
    """
    Generate n_frames for a Ken Burns (zoom in/out) effect.

    Parameters
    ----------
    base_arr : ndarray (H, W, 3)
    n_frames : int
    direction : 'zoom_in' or 'zoom_out'
    zoom_factor : final zoom level (1.12 = 12% zoom)
    """
    H, W = base_arr.shape[:2]
    frames = []

    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)   # 0.0 → 1.0
        if direction == "zoom_in":
            scale = 1.0 + (zoom_factor - 1.0) * t
        else:
            scale = zoom_factor - (zoom_factor - 1.0) * t

        new_w = int(W / scale)
        new_h = int(H / scale)
        x0 = (W - new_w) // 2
        y0 = (H - new_h) // 2

        crop = base_arr[y0:y0 + new_h, x0:x0 + new_w]
        img = Image.fromarray(crop).resize((W, H), Image.LANCZOS)
        frames.append(np.array(img))

    return frames


def _pan_frames(
    base_arr: np.ndarray,
    n_frames: int,
    dx: int = 0,
    dy: int = 0,
    zoom_factor: float = MOTION_ZOOM_FACTOR,
) -> List[np.ndarray]:
    """
    Generate frames for a pan effect. Image is pre-zoomed so panning
    stays within bounds.

    Parameters
    ----------
    dx : horizontal pixels to shift total (positive = pan right)
    dy : vertical pixels to shift total (positive = pan down)
    """
    H, W = base_arr.shape[:2]
    # Pre-zoom the image
    zoomed_w = int(W * zoom_factor)
    zoomed_h = int(H * zoom_factor)
    zoomed = np.array(Image.fromarray(base_arr).resize((zoomed_w, zoomed_h), Image.LANCZOS))

    # Starting crop position (centered)
    start_x = (zoomed_w - W) // 2
    start_y = (zoomed_h - H) // 2

    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        cx = start_x + int(dx * t)
        cy = start_y + int(dy * t)
        # Clamp
        cx = max(0, min(cx, zoomed_w - W))
        cy = max(0, min(cy, zoomed_h - H))
        crop = zoomed[cy:cy + H, cx:cx + W]
        frames.append(crop)

    return frames


def _static_frames(base_arr: np.ndarray, n_frames: int) -> List[np.ndarray]:
    """All frames identical — no motion."""
    return [base_arr.copy() for _ in range(n_frames)]


def _camera_motion_to_frames(
    base_arr: np.ndarray,
    n_frames: int,
    camera_motion: str,
) -> List[np.ndarray]:
    """Route camera_motion string to the correct frame generator."""
    motion = (camera_motion or "Static").lower().strip()
    H, W = base_arr.shape[:2]
    pan_pixels = int(min(W, H) * 0.08)   # 8% of shorter dimension

    if motion == "zoom in":
        return _ken_burns_frames(base_arr, n_frames, "zoom_in")
    elif motion == "zoom out":
        return _ken_burns_frames(base_arr, n_frames, "zoom_out")
    elif motion == "pan left":
        return _pan_frames(base_arr, n_frames, dx=-pan_pixels, dy=0)
    elif motion == "pan right":
        return _pan_frames(base_arr, n_frames, dx=+pan_pixels, dy=0)
    elif motion == "tilt up":
        return _pan_frames(base_arr, n_frames, dx=0, dy=-pan_pixels)
    elif motion == "tilt down":
        return _pan_frames(base_arr, n_frames, dx=0, dy=+pan_pixels)
    elif motion in ("tracking shot", "diagonal pan"):
        return _pan_frames(base_arr, n_frames, dx=pan_pixels, dy=-pan_pixels // 2)
    else:
        return _static_frames(base_arr, n_frames)


def _crossfade(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    n_frames: int,
) -> List[np.ndarray]:
    """Blend two frames over n_frames."""
    frames = []
    for i in range(n_frames):
        alpha = i / max(n_frames - 1, 1)
        blended = ((1 - alpha) * arr_a.astype(float) + alpha * arr_b.astype(float)).astype(np.uint8)
        frames.append(blended)
    return frames


class MotionEngine:
    """
    Converts storyboard images into motion video clips.

    Each clip duration = max(narration_duration, MIN_SCENE_DURATION)
    so video stays synchronized with TTS audio.
    """

    def __init__(
        self,
        project_name: str,
        projects_root: str = "projects",
        fps: int = DEFAULT_FPS,
        resolution: str = "1080p",
    ):
        self.project_name = project_name
        self.project_dir = Path(projects_root) / project_name
        self.fps = fps
        self.resolution = resolution

        # Parse target size
        from lib.image_generator import RESOLUTION_MAP
        h, w = RESOLUTION_MAP.get(resolution, (1080, 1920))
        self.target_size = (w, h)  # PIL (width, height)

        self.motion_cache_dir = self.project_dir / "cache" / "motion"
        self.motion_cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_clip_path(self, segment_id: str) -> Path:
        return self.motion_cache_dir / f"scene_{segment_id}.mp4"

    def render_scene(
        self,
        segment_id: str,
        image_path: Path,
        camera_motion: str,
        narration_duration: float,
        transition: str = "cut",
    ) -> Path:
        """
        Render one scene clip.

        Parameters
        ----------
        narration_duration : float
            Actual audio duration in seconds. Clip will be at least this long.
        transition : str
            'cut', 'fade', or 'dissolve'

        Returns
        -------
        Path to the rendered .mp4 clip
        """
        out_path = self._get_clip_path(segment_id)
        if out_path.exists():
            logger.debug(f"  {segment_id}: motion clip exists, skipping")
            return out_path

        # Dynamic duration — never shorter than narration
        duration = max(float(narration_duration), MIN_SCENE_DURATION)
        n_frames = math.ceil(duration * self.fps)

        logger.info(f"  {segment_id}: {camera_motion} effect, {duration:.1f}s, {n_frames} frames")

        base_arr = _load_image_as_array(image_path, self.target_size)
        frames = _camera_motion_to_frames(base_arr, n_frames, camera_motion)

        # Add fade-out for dissolve transitions (last 0.5s)
        if transition in ("fade", "dissolve"):
            fade_frames = int(0.5 * self.fps)
            black = np.zeros_like(base_arr)
            fade = _crossfade(frames[-1], black, fade_frames)
            frames = frames[:-fade_frames] + fade if len(frames) > fade_frames else frames

        self._write_video(frames, out_path)
        return out_path

    def _write_video(self, frames: List[np.ndarray], out_path: Path) -> None:
        """Write frame list to MP4 using MoviePy."""
        try:
            from moviepy import ImageSequenceClip
            clip = ImageSequenceClip(
                [f for f in frames],  # list of numpy arrays
                fps=self.fps,
            )
            clip.write_videofile(
                str(out_path),
                codec="libx264",
                audio=False,
                logger=None,    # suppress moviepy verbose output
                ffmpeg_params=["-crf", "23", "-preset", "fast"],
            )
        except ImportError:
            # Fallback: write via OpenCV
            import cv2
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (w, h),
            )
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

    def render_episode(
        self,
        episode: int,
        audio_durations: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Render motion clips for all scenes in an episode.

        Parameters
        ----------
        audio_durations : dict
            {segment_id: duration_seconds} from TTS output.
            If None, uses storyboard duration_seconds field.
        """
        script_path = self.project_dir / "scripts" / f"episode_{episode}.json"
        if not script_path.exists():
            raise FileNotFoundError(f"Missing storyboard: {script_path}")

        with open(script_path, encoding="utf-8") as f:
            script = json.load(f)

        segments = script.get("segments", [])
        total = len(segments)
        rendered = 0
        skipped = 0
        failed = 0
        clip_paths: Dict[str, str] = {}

        for seg in segments:
            sid = seg["segment_id"]
            out_path = self._get_clip_path(sid)

            if out_path.exists():
                clip_paths[sid] = str(out_path)
                skipped += 1
                continue

            assets = seg.get("generated_assets", {})
            image_path_str = assets.get("storyboard_image")
            if not image_path_str or not Path(image_path_str).exists():
                logger.warning(f"  {sid}: no image found, skipping motion")
                failed += 1
                continue

            # Duration from audio > storyboard field
            narr_duration = (
                (audio_durations or {}).get(sid)
                or seg.get("duration_seconds", 4)
            )

            try:
                clip_path = self.render_scene(
                    segment_id=sid,
                    image_path=Path(image_path_str),
                    camera_motion=seg.get("camera_motion", "Static"),
                    narration_duration=narr_duration,
                    transition=seg.get("transition_to_next", "cut"),
                )
                clip_paths[sid] = str(clip_path)
                rendered += 1
            except Exception as e:
                logger.error(f"  {sid}: motion render failed: {e}")
                failed += 1

        logger.info(f"Motion engine: {rendered} rendered, {skipped} skipped, {failed} failed / {total} total")
        return {
            "rendered": rendered,
            "skipped": skipped,
            "failed": failed,
            "total": total,
            "clip_paths": clip_paths,
        }

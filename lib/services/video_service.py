"""
lib/services/video_service.py — Motion engine + video composition service.

Replaces: generation_tasks.py execute_video_task() (which called Veo API).
Now calls: lib/motion_engine.py + lib/video_composer.py (all local, no API).

Callable from WebUI, CLI, and Kaggle notebook identically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def render_episode_motion(
    project_name: str,
    episode: int,
    audio_durations: Optional[Dict[str, float]] = None,
    projects_root: Optional[str] = None,
    resolution: str = "1080p",
    fps: int = 24,
) -> Dict[str, Any]:
    """
    Render motion video clips (Ken Burns, pan, etc.) for all scenes in an episode.

    Each clip duration = max(narration_duration, MIN_SCENE_DURATION).
    No AI video model required.

    Called from:
      - WebUI: POST /projects/{name}/pipeline/motion
      - CLI:   run_pipeline.py (between Stage 2 and 4)
      - Notebook: video_service.render_episode_motion(...)
    """
    from lib.motion_engine import MotionEngine

    project_dir = _get_project_dir(project_name, projects_root)
    engine = MotionEngine(
        project_name=project_name,
        projects_root=str(project_dir.parent),
        fps=fps,
        resolution=resolution,
    )
    return engine.render_episode(episode, audio_durations=audio_durations)


def compose_episode_video(
    project_name: str,
    episode: int,
    projects_root: Optional[str] = None,
    resolution: str = "1080p",
    include_subtitles: bool = True,
    bgm_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compose the final episode video from motion clips + audio + subtitles.

    Called from:
      - WebUI: POST /projects/{name}/pipeline/compose
      - CLI:   run_pipeline.py Stage 4
      - Notebook: video_service.compose_episode_video(...)
    """
    from lib.video_composer import VideoComposer

    project_dir = _get_project_dir(project_name, projects_root)
    composer = VideoComposer(
        project_name=project_name,
        projects_root=str(project_dir.parent),
        resolution=resolution,
    )
    output_path = composer.compose_episode(
        episode=episode,
        include_subtitles=include_subtitles,
        bgm_path=bgm_path,
    )
    return {
        "output_path": str(output_path),
        "episode": episode,
        "resolution": resolution,
    }


def render_single_scene_motion(
    project_name: str,
    segment_id: str,
    image_path: str,
    camera_motion: str,
    narration_duration: float,
    projects_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Render motion clip for a single scene (for WebUI preview).

    Replaces the Veo video generation call for single-segment preview.
    """
    from lib.motion_engine import MotionEngine
    from pathlib import Path as _Path

    project_dir = _get_project_dir(project_name, projects_root)
    engine = MotionEngine(
        project_name=project_name,
        projects_root=str(project_dir.parent),
    )
    clip_path = engine.render_scene(
        segment_id=segment_id,
        image_path=_Path(image_path),
        camera_motion=camera_motion,
        narration_duration=narration_duration,
    )
    return {
        "segment_id": segment_id,
        "clip_path": str(clip_path),
        "resource_type": "motion_clips",
    }


def _get_project_dir(project_name: str, projects_root: Optional[str]) -> Path:
    root = Path(projects_root) if projects_root else Path("projects")
    return root / project_name

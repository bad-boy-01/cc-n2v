"""
lib/services/audio_service.py — TTS audio synthesis service.

Replaces any Veo/TTS calls inside generation_tasks.py.
Callable from WebUI, CLI, and Kaggle notebook identically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def synthesize_episode_audio(
    project_name: str,
    episode: int,
    projects_root: Optional[str] = None,
    voice: str = "af_heart",
    speed: float = 1.0,
) -> Dict[str, Any]:
    """
    Synthesize TTS audio for all segments in an episode.

    Called from:
      - WebUI: POST /projects/{name}/pipeline/audio
      - CLI:   run_pipeline.py Stage 3
      - Notebook: audio_service.synthesize_episode_audio(...)

    Returns
    -------
    dict: audio_dir, segment_count, durations {segment_id: seconds}
    """
    from lib.kokoro_tts import KokoroTTS

    project_dir = _get_project_dir(project_name, projects_root)
    tts = KokoroTTS(
        project_name=project_name,
        projects_root=str(project_dir.parent),
        voice=voice,
        speed=speed,
    )
    result = tts.synthesize_episode(episode)
    tts.unload_model()
    return result


def _get_project_dir(project_name: str, projects_root: Optional[str]) -> Path:
    root = Path(projects_root) if projects_root else Path("projects")
    return root / project_name

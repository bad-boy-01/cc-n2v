"""
lib/services/pipeline_service.py — Full pipeline orchestrator service.

THE single entry point for running the complete pipeline.
Identical call signature from WebUI, CLI, and Kaggle notebook.

Usage
-----
# From Python anywhere:
from lib.services.pipeline_service import run_pipeline

run_pipeline(
    project_name="my_novel",
    input_file="source/novel.txt",
    mode="novel",
    episode=1,
)

# From CLI:
# python run_pipeline.py --input novel.txt --mode novel

# From WebUI:
# POST /api/v1/projects/{name}/pipeline/run
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from lib.config import DEFAULT_LLM

logger = logging.getLogger(__name__)


def _free_vram() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_pipeline(
    project_name: str,
    input_file: Optional[str] = None,
    mode: str = "novel",
    episode: int = 1,
    projects_root: Optional[str] = None,
    llm: str = DEFAULT_LLM,
    load_in_4bit: bool = True,
    resolution: str = "1080p",
    storyboard_only: bool = False,
    dry_run: bool = False,
    resume: bool = True,
    fast_cpu_overlap: bool = False,
    max_scenes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the complete video generation pipeline.

    Enforces staged model loading:
      Stage 1 (Qwen)  → Stage 2 (FLUX) → Stage 3 (Kokoro) → Stage 4 (FFmpeg)

    Parameters
    ----------
    mode : str
        "novel"            — full novel pipeline
        "manhwa"           — OCR → summarize → regenerate images
        "manhwa_panels"    — panel detection → animate original panels
        "storyboard_only"  — stop after storyboard JSON generation
    resume : bool
        If True (default), skip already-completed stages.
    max_scenes : int, optional
        If set, process only the first N storyboard scenes (debug mode).

    Returns
    -------
    dict with output path and per-stage results
    """
    from lib.agents.video_agent import VideoAgent

    agent = VideoAgent(
        project_name=project_name,
        episode=episode,
        projects_root=projects_root,
        input_file=input_file,
        mode=mode,
        llm=llm,
        load_in_4bit=load_in_4bit,
        resolution=resolution,
        storyboard_only=storyboard_only,
        dry_run=dry_run,
        fast_cpu_overlap=fast_cpu_overlap,
        max_scenes=max_scenes,
    )

    return agent.run()



def run_storyboard_stage(
    project_name: str,
    episode: int = 1,
    projects_root: Optional[str] = None,
    llm: str = DEFAULT_LLM,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Run ONLY Stage 1 (LLM storyboard generation).
    Unloads Qwen when done. Safe to call before Stage 2.
    """
    from lib.services.storyboard_service import run_storyboard_pipeline
    from lib.qwen_client import get_qwen_client

    result = run_storyboard_pipeline(
        project_name=project_name,
        episode=episode,
        llm=llm,
        overwrite=overwrite,
        projects_root=projects_root,
    )

    # Always unload after storyboard stage
    try:
        client = get_qwen_client(llm)
        client.unload_model()
    except Exception:
        pass
    _free_vram()

    return result


def run_image_stage(
    project_name: str,
    episode: int = 1,
    projects_root: Optional[str] = None,
    resolution: str = "1080p",
    dry_run: bool = False,
    max_scenes: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ONLY Stage 2 (FLUX image generation). Unloads FLUX when done."""
    from lib.services.image_service import generate_episode_images, generate_all_character_portraits

    # Generate character portraits first (while FLUX is loaded)
    portrait_result = generate_all_character_portraits(project_name, projects_root)

    # Generate episode images
    image_result = generate_episode_images(
        project_name=project_name,
        episode=episode,
        projects_root=projects_root,
        resolution=resolution,
        dry_run=dry_run,
        max_scenes=max_scenes,
    )
    _free_vram()

    return {"portraits": portrait_result, "images": image_result}


def run_audio_stage(
    project_name: str,
    episode: int = 1,
    projects_root: Optional[str] = None,
    voice: str = "af_heart",
) -> Dict[str, Any]:
    """Run ONLY Stage 3 (Kokoro TTS). Unloads Kokoro when done."""
    from lib.services.audio_service import synthesize_episode_audio

    result = synthesize_episode_audio(
        project_name=project_name,
        episode=episode,
        projects_root=projects_root,
        voice=voice,
    )
    _free_vram()
    return result


def run_compose_stage(
    project_name: str,
    episode: int = 1,
    projects_root: Optional[str] = None,
    resolution: str = "1080p",
    include_subtitles: bool = True,
    bgm_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run ONLY Stage 4 (FFmpeg compose). CPU only — no model loading."""
    from lib.services.video_service import render_episode_motion, compose_episode_video
    from lib.services.audio_service import synthesize_episode_audio

    # Get audio durations for dynamic image duration
    audio_dir = Path(projects_root or "projects") / project_name / "audio"
    durations: Dict[str, float] = {}
    for duration_file in audio_dir.glob("*.duration"):
        seg_id = duration_file.stem
        try:
            durations[seg_id] = float(duration_file.read_text().strip())
        except Exception:
            pass

    # Render motion clips with audio-synced durations
    render_episode_motion(
        project_name=project_name,
        episode=episode,
        audio_durations=durations or None,
        projects_root=projects_root,
        resolution=resolution,
    )

    # Compose final video
    return compose_episode_video(
        project_name=project_name,
        episode=episode,
        projects_root=projects_root,
        resolution=resolution,
        include_subtitles=include_subtitles,
        bgm_path=bgm_path,
    )


def get_pipeline_state(
    project_name: str,
    projects_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Return the current pipeline state (project_state.json).
    Used by WebUI to show progress and by notebook for resume decisions.
    """
    from lib.agents.video_agent import PipelineState
    from lib.project_manager import ProjectManager

    root = Path(projects_root) if projects_root else Path("projects")
    pm = ProjectManager(root)
    project_dir = pm.get_project_path(project_name)
    state = PipelineState(project_dir)
    return state._state

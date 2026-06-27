"""
lib/services/storyboard_service.py — Storyboard generation service.

The single source of truth for storyboard operations.
Called identically from: WebUI router, CLI, Kaggle notebook.

Replaces: webui/server/services/generation_tasks.py → execute_storyboard_task()
          (which called GeminiClient directly)

Now calls: lib/agents/storyboard_agent.py → lib/qwen_client.py (local)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.config import DEFAULT_LLM
from lib.project_manager import ProjectManager

logger = logging.getLogger(__name__)

_pm: Optional[ProjectManager] = None


def _get_pm(projects_root: Optional[str] = None) -> ProjectManager:
    global _pm
    if _pm is None or projects_root:
        root = Path(projects_root) if projects_root else Path("projects")
        _pm = ProjectManager(root)
    return _pm


# ── Public service functions (callable from anywhere) ─────────────────────────

def run_storyboard_pipeline(
    project_name: str,
    episode: int = 1,
    llm: str = DEFAULT_LLM,
    load_in_4bit: bool = True,
    overwrite: bool = False,
    projects_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full LLM storyboard pipeline for one episode.

    Pipeline:
      StoryAgent → CharacterAgent → ClueAgent → StoryboardAgent → NarrationAgent

    Called from:
      - WebUI: POST /projects/{name}/pipeline/storyboard
      - CLI:   python run_pipeline.py --mode storyboard_only
      - Notebook: from lib.services.storyboard_service import run_storyboard_pipeline

    Returns
    -------
    dict with: output_path, total_scenes, characters_found, clues_found
    """
    from lib.agents.story_agent import StoryAgent
    from lib.agents.character_agent import CharacterAgent
    from lib.agents.clue_agent import ClueAgent
    from lib.agents.storyboard_agent import StoryboardAgent
    from lib.agents.narration_agent import NarrationAgent

    agent_kwargs = dict(
        projects_root=projects_root,
        llm=llm,
        load_in_4bit=load_in_4bit,
    )

    logger.info(f"[storyboard_service] Episode {episode} — starting pipeline")

    story = StoryAgent(project_name, episode=episode, **agent_kwargs)
    story_result = story.run(overwrite=overwrite)

    char = CharacterAgent(project_name, episode=episode, **agent_kwargs)
    char_result = char.run(overwrite=overwrite)

    clue = ClueAgent(project_name, episode=episode, **agent_kwargs)
    clue_result = clue.run(overwrite=overwrite)

    sb = StoryboardAgent(project_name, episode=episode, **agent_kwargs)
    sb_result = sb.run(overwrite=overwrite)

    narr = NarrationAgent(project_name, episode=episode, **agent_kwargs)
    narr_result = narr.run(overwrite=overwrite)

    return {
        "project": project_name,
        "episode": episode,
        "output_path": sb_result.get("output_path"),
        "total_scenes": sb_result.get("total_scenes", 0),
        "total_segments": story_result.get("total_segments", 0),
        "characters_found": char_result.get("character_count", 0),
        "clues_found": clue_result.get("clue_count", 0),
        "estimated_minutes": story_result.get("estimated_minutes", 0),
        "storyboard_complete": sb_result.get("complete", False),
    }


def generate_single_scene_storyboard(
    project_name: str,
    segment_id: str,
    script_file: str,
    prompt: Any,
    extra_reference_images: Optional[List[str]] = None,
    projects_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate or regenerate the storyboard image for one scene.

    Maintains backward compatibility with the existing WebUI queue task format.
    Called from: webui/server/services/generation_tasks.py (redirected here).

    Parameters
    ----------
    prompt : str | dict
        Raw prompt string or structured prompt dict from WebUI.
    """
    from lib.image_generator import ImageGenerator
    from lib.services._prompt_utils import normalize_storyboard_prompt, collect_reference_images

    pm = _get_pm(projects_root)
    project = pm.load_project(project_name)
    project_path = pm.get_project_path(project_name)
    script = pm.load_script(project_name, script_file)

    # Find the target segment
    items, id_field, char_field, clue_field = _get_items_from_script(script)
    target = next((item for item in items if str(item.get(id_field)) == segment_id), None)
    if not target:
        raise ValueError(f"Segment not found: {segment_id}")

    style = project.get("style", "")
    prompt_text = normalize_storyboard_prompt(prompt, style)
    ref_images = collect_reference_images(
        project, project_path, target,
        char_field=char_field, clue_field=clue_field,
        extra_reference_images=extra_reference_images,
    )

    gen = ImageGenerator(
        project_name=project_name,
        projects_root=str(project_path.parent),
    )

    # Build minimal scene dict for the generator
    scene = {
        "segment_id": segment_id,
        "location": target.get("location", ""),
        "characters": target.get(char_field, []),
        "scene_type": target.get("scene_type", "exposition"),
        "image_prompt": prompt_text,
        "camera_motion": target.get("camera_motion", "Static"),
    }

    image_path, backend = gen._generate_image(scene)
    if image_path is None:
        raise RuntimeError(f"Image generation failed for {segment_id}")

    pm.update_scene_asset(
        project_name=project_name,
        script_filename=script_file,
        scene_id=segment_id,
        asset_type="storyboard_image",
        asset_path=f"images/scene_{segment_id}.png",
    )

    logger.info(f"[storyboard_service] {segment_id}: generated via {backend} → {image_path}")
    return {
        "segment_id": segment_id,
        "file_path": str(image_path),
        "backend": backend,
        "resource_type": "storyboards",
    }


def _get_items_from_script(script: dict):
    """Shared utility: return (items, id_field, char_field, clue_field)."""
    content_mode = script.get("content_mode", "narration")
    if content_mode == "narration" and "segments" in script:
        return script["segments"], "segment_id", "characters", "characters_in_segment"
    return script.get("scenes", []), "scene_id", "characters_in_scene", "clues_in_scene"

"""
lib/services/image_service.py — Image generation service.

Single entry point for all image generation operations.
Replaces: generation_tasks.py execute_character_task() + execute_clue_task()

Called identically from WebUI, CLI, and Kaggle notebook.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_episode_images(
    project_name: str,
    episode: int,
    projects_root: Optional[str] = None,
    resolution: str = "1080p",
    dry_run: bool = False,
    max_scenes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate all storyboard images for an episode.

    Handles scene cache lookup, scene-level checkpointing, and
    FLUX → SDXL → reuse fallback chain automatically.

    Called from:
      - WebUI: POST /projects/{name}/pipeline/images
      - CLI:   run_pipeline.py Stage 2
      - Notebook: image_service.generate_episode_images(...)

    Parameters
    ----------
    max_scenes : int, optional
        If set, process only the first N scenes (debug/fast-iteration mode).
    """
    from lib.image_generator import ImageGenerator

    project_dir = _get_project_dir(project_name, projects_root)
    gen = ImageGenerator(
        project_name=project_name,
        projects_root=str(project_dir.parent),
        resolution=resolution,
        dry_run=dry_run,
    )
    result = gen.generate_episode(episode, max_scenes=max_scenes)
    gen.unload_model()
    return result


def generate_character_portrait(
    project_name: str,
    character_name: str,
    projects_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a character reference portrait (hero_ref.png).

    Replaces: generation_tasks.py execute_character_task() Gemini call.

    The portrait is generated using the existing hero.json + hero_prompt.txt.
    Called during Stage 2 so Qwen is already unloaded.
    """
    from lib.image_generator import ImageGenerator

    project_dir = _get_project_dir(project_name, projects_root)
    stem = _to_stem(character_name)
    json_path = project_dir / "characters" / f"{stem}.json"

    char_data = {}
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            char_data = json.load(f)

    gen = ImageGenerator(
        project_name=project_name,
        projects_root=str(project_dir.parent),
    )
    ref_path = gen.generate_character_portrait(character_name, char_data)
    gen.unload_model()

    if ref_path is None:
        raise RuntimeError(f"Failed to generate portrait for: {character_name}")

    logger.info(f"[image_service] Character portrait: {ref_path}")
    return {
        "character": character_name,
        "file_path": str(ref_path),
        "resource_type": "characters",
    }


def generate_all_character_portraits(
    project_name: str,
    projects_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate reference portraits for all characters that don't have one yet.
    Batches portrait generation into one FLUX session for efficiency.
    """
    from lib.image_generator import ImageGenerator

    project_dir = _get_project_dir(project_name, projects_root)
    chars_dir = project_dir / "characters"

    if not chars_dir.exists():
        return {"generated": 0, "skipped": 0}

    gen = ImageGenerator(
        project_name=project_name,
        projects_root=str(project_dir.parent),
    )

    generated = 0
    skipped = 0

    for json_file in sorted(chars_dir.glob("*.json")):
        stem = json_file.stem
        ref_path = chars_dir / f"{stem}_ref.png"
        if ref_path.exists():
            skipped += 1
            continue

        with open(json_file, encoding="utf-8") as f:
            char_data = json.load(f)

        char_name = char_data.get("name", stem.replace("_", " ").title())
        result = gen.generate_character_portrait(char_name, char_data)
        if result:
            generated += 1
            logger.info(f"  {char_name}: portrait generated")
        else:
            logger.warning(f"  {char_name}: portrait generation failed")

    gen.unload_model()
    return {"generated": generated, "skipped": skipped}


def generate_clue_image(
    project_name: str,
    clue_name: str,
    prompt: str,
    projects_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an image for a clue/location.

    Replaces: generation_tasks.py execute_clue_task() Gemini call.
    """
    from lib.image_generator import ImageGenerator
    from lib.project_manager import ProjectManager

    project_dir = _get_project_dir(project_name, projects_root)
    pm = ProjectManager(project_dir.parent)
    project = pm.load_project(project_name)

    if clue_name not in project.get("clues", {}):
        raise ValueError(f"Clue not found: {clue_name}")

    clue_data = project["clues"][clue_name]
    clue_type = clue_data.get("type", "location")
    style = project.get("style", "")
    full_prompt = f"{prompt}, {clue_type}, {style}".strip(", ")

    gen = ImageGenerator(
        project_name=project_name,
        projects_root=str(project_dir.parent),
    )

    stem = _to_stem(clue_name)
    fake_scene = {
        "segment_id": f"clue_{stem}",
        "location": clue_name,
        "characters": [],
        "scene_type": "exterior" if clue_type == "location" else "exposition",
        "image_prompt": full_prompt,
    }

    image_path, backend = gen._generate_image(fake_scene)
    gen.unload_model()

    if image_path is None:
        raise RuntimeError(f"Clue image generation failed: {clue_name}")

    # Update project.json
    project["clues"][clue_name]["clue_sheet"] = f"clues/{stem}.png"
    pm.save_project(project_name, project)

    return {
        "clue": clue_name,
        "file_path": str(image_path),
        "backend": backend,
        "resource_type": "clues",
    }


def _get_project_dir(project_name: str, projects_root: Optional[str]) -> Path:
    root = Path(projects_root) if projects_root else Path("projects")
    return root / project_name


def _to_stem(name: str) -> str:
    import re
    return re.sub(r"[^\w\-]", "_", name.strip()).lower()

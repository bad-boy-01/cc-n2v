"""
Task execution service dispatcher.

Delegates ALL business logic to lib/services/ functions,
making the same logic available from WebUI, CLI, and Kaggle notebook.

The existing generation_queue.py + generation_worker.py call
execute_generation_task() here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from lib.project_manager import ProjectManager

logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent.parent
pm = ProjectManager(project_root / "projects")


def execute_storyboard_task(
    project_name: str, resource_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate storyboard image for one scene.
    Delegated to: lib/services/storyboard_service.generate_single_scene_storyboard()
    """
    from lib.services.storyboard_service import generate_single_scene_storyboard

    script_file = payload.get("script_file")
    if not script_file:
        raise ValueError("script_file is required")

    prompt = payload.get("prompt")
    if prompt is None:
        raise ValueError("prompt is required")

    return generate_single_scene_storyboard(
        project_name=project_name,
        segment_id=resource_id,
        script_file=script_file,
        prompt=prompt,
        extra_reference_images=payload.get("extra_reference_images") or [],
        projects_root=str(project_root / "projects"),
    )


def execute_video_task(
    project_name: str, resource_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate motion video for one scene.
    Delegated to: lib/services/video_service.render_single_scene_motion()
    """
    from lib.services.video_service import render_single_scene_motion

    script_file = payload.get("script_file")
    if not script_file:
        raise ValueError("script_file is required")

    project_path = pm.get_project_path(project_name)
    storyboard_file = project_path / "images" / f"scene_{resource_id}.png"
    if not storyboard_file.exists():
        # Fallback: old storyboards/ path
        storyboard_file = project_path / "storyboards" / f"scene_{resource_id}.png"
    if not storyboard_file.exists():
        raise ValueError(f"Storyboard not found: scene_{resource_id}.png")

    # Get camera motion from script
    camera_motion = "Static"
    narration_duration = float(payload.get("duration_seconds") or 4)
    try:
        script = pm.load_script(project_name, script_file)
        for item in script.get("segments", script.get("scenes", [])):
            id_field = "segment_id" if "segments" in script else "scene_id"
            if str(item.get(id_field)) == resource_id:
                camera_motion = item.get("camera_motion", "Static")
                narration_duration = float(item.get("duration_seconds", 4))
                break
    except Exception:
        pass

    return render_single_scene_motion(
        project_name=project_name,
        segment_id=resource_id,
        image_path=str(storyboard_file),
        camera_motion=camera_motion,
        narration_duration=narration_duration,
        projects_root=str(project_root / "projects"),
    )


def execute_character_task(
    project_name: str, resource_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate character reference portrait.
    Delegated to: lib/services/image_service.generate_character_portrait()
    """
    from lib.services.image_service import generate_character_portrait

    return generate_character_portrait(
        project_name=project_name,
        character_name=resource_id,
        projects_root=str(project_root / "projects"),
    )


def execute_clue_task(
    project_name: str, resource_id: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate clue/location image.
    Delegated to: lib/services/image_service.generate_clue_image()
    """
    from lib.services.image_service import generate_clue_image

    prompt = str(payload.get("prompt", "") or "").strip()
    if not prompt:
        raise ValueError("prompt is required for clue task")

    return generate_clue_image(
        project_name=project_name,
        clue_name=resource_id,
        prompt=prompt,
        projects_root=str(project_root / "projects"),
    )


def execute_storyboard_grid_task(
    project_name: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a storyboard grid (drama mode).
    In narration mode (the default), this is unused.
    Falls back to generating individual images via image_service.
    """
    scene_ids = payload.get("scene_ids") or []
    batch_id = payload.get("batch_id", 0)
    script_file = payload.get("script_file", "")

    if not scene_ids:
        raise ValueError("scene_ids must be a non-empty list")

    # In free mode: generate individual images for each scene in the batch
    from lib.services.image_service import generate_character_portrait
    results = []
    for scene_id in scene_ids:
        try:
            result = execute_storyboard_task(
                project_name,
                scene_id,
                {"script_file": script_file, "prompt": ""},
            )
            results.append(result)
        except Exception as e:
            logger.warning(f"Scene {scene_id} in grid batch failed: {e}")

    return {
        "batch_id": batch_id,
        "scene_ids": scene_ids,
        "results": results,
        "resource_type": "storyboard_grid",
        "resource_id": f"batch_{batch_id}",
    }


def execute_generation_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main task dispatcher — called by generation_worker.py.
    """
    task_type = task.get("task_type")
    project_name = task.get("project_name")
    resource_id = task.get("resource_id")
    payload = task.get("payload") or {}

    if not project_name:
        raise ValueError("task.project_name is required")

    if task_type == "storyboard":
        return execute_storyboard_task(project_name, str(resource_id), payload)
    if task_type == "video":
        return execute_video_task(project_name, str(resource_id), payload)
    if task_type == "character":
        return execute_character_task(project_name, str(resource_id), payload)
    if task_type == "clue":
        return execute_clue_task(project_name, str(resource_id), payload)
    if task_type == "storyboard_grid":
        return execute_storyboard_grid_task(project_name, payload)

    raise ValueError(f"Unsupported task_type: {task_type}")

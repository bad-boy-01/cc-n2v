"""
lib/services/_prompt_utils.py — Shared prompt normalization utilities.

Extracted from: webui/server/services/generation_tasks.py
Replaces the GeminiClient-specific prompt helpers with backend-agnostic versions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def normalize_storyboard_prompt(prompt: Union[str, dict], style: str = "") -> str:
    """
    Normalize a raw prompt (string or structured dict) to a plain text string
    suitable for FLUX/SDXL image generation.

    Handles the existing WebUI structured prompt format for backward compatibility.
    """
    if isinstance(prompt, str):
        return prompt.strip()

    if not isinstance(prompt, dict):
        raise ValueError("prompt must be a string or dict")

    scene = str(prompt.get("scene", "")).strip()
    if not scene:
        raise ValueError("prompt.scene must not be empty")

    parts = [scene]
    composition = prompt.get("composition", {}) or {}
    if isinstance(composition, dict):
        if composition.get("shot_type"):
            parts.append(str(composition["shot_type"]))
        if composition.get("lighting"):
            parts.append(str(composition["lighting"]))
        if composition.get("ambiance"):
            parts.append(str(composition["ambiance"]))

    if style:
        parts.append(style)

    return ", ".join(p for p in parts if p)


def normalize_video_prompt(prompt: Union[str, dict]) -> str:
    """
    Normalize video prompt to plain text.
    Previously this produced YAML for Veo — now just returns action description.
    """
    if isinstance(prompt, str):
        return prompt.strip()

    if not isinstance(prompt, dict):
        raise ValueError("prompt must be a string or dict")

    action = str(prompt.get("action", "")).strip()
    if not action:
        raise ValueError("prompt.action must not be empty")

    camera = prompt.get("camera_motion", "Static")
    return f"{action}. Camera: {camera}"


def collect_reference_images(
    project: dict,
    project_path: Path,
    target_item: dict,
    *,
    char_field: str,
    clue_field: str,
    extra_reference_images: Optional[List[str]] = None,
) -> List[Path]:
    """
    Collect character_sheet and clue_sheet reference images for a scene.

    Migrated from generation_tasks.py _collect_reference_images().
    Now also checks for the new {stem}_ref.png format from character_agent.
    """
    reference_images: List[Path] = []

    # Check new-style ref images (hero_ref.png) first
    chars_dir = project_path / "characters"
    for char_name in target_item.get(char_field, []):
        stem = re.sub(r"[^\w\-]", "_", char_name.strip()).lower()
        new_ref = chars_dir / f"{stem}_ref.png"
        if new_ref.exists():
            reference_images.append(new_ref)
            continue

        # Fallback: old character_sheet format
        char_data = project.get("characters", {}).get(char_name, {})
        sheet = char_data.get("character_sheet")
        if sheet:
            path = project_path / sheet
            if path.exists():
                reference_images.append(path)

    for clue_name in target_item.get(clue_field, []):
        clue_data = project.get("clues", {}).get(clue_name, {})
        sheet = clue_data.get("clue_sheet")
        if sheet:
            path = project_path / sheet
            if path.exists():
                reference_images.append(path)

    for extra in extra_reference_images or []:
        extra_path = Path(extra)
        if not extra_path.is_absolute():
            extra_path = project_path / extra_path
        if extra_path.exists():
            reference_images.append(extra_path)

    return reference_images


def normalize_duration(duration_seconds: Any) -> int:
    """Normalize a duration value to 4, 6, or 8 seconds."""
    try:
        value = int(duration_seconds) if duration_seconds is not None else 4
    except (TypeError, ValueError):
        value = 4
    if value <= 4:
        return 4
    if value <= 6:
        return 6
    return 8

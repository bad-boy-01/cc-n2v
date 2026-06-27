"""
lib/prompt_builder.py — Structured image prompt assembler.

Takes a scene dict (from StoryboardAgent) and assembles a rich, complete
prompt for FLUX/SDXL by injecting:
  1. Scene base description
  2. Character visual state (from WorldEngine)
  3. Location atmosphere (from WorldEngine)
  4. Time-of-day lighting keywords
  5. Camera/shot type keywords
  6. Emotion-based mood keywords
  7. Art style (from DirectorProfile)

This ensures every image prompt is complete, consistent, and style-aware
without requiring duplicate information in the storyboard JSON.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Time-of-day lighting keyword tables ───────────────────────────────────────

TIME_LIGHTING: Dict[str, str] = {
    "dawn": "golden hour, warm sunrise light, long shadows, soft pink sky",
    "morning": "bright morning light, fresh atmosphere, clear sky",
    "day": "natural daylight, soft shadows, clear lighting",
    "midday": "harsh midday sun, high contrast shadows, bright daylight",
    "afternoon": "warm afternoon light, golden tone",
    "dusk": "twilight, purple-orange sky, dramatic atmosphere, fading light",
    "sunset": "golden sunset, warm glow, long shadows",
    "evening": "dim evening light, warm indoor glow, blue hour",
    "night": "moonlit, deep shadows, star-filled sky, cool blue tones",
    "midnight": "deep night, darkness, silvery moonlight, dramatic shadows",
}

# ── Camera/shot keyword tables ────────────────────────────────────────────────

CAMERA_KEYWORDS: Dict[str, str] = {
    "Static":        "medium shot",
    "Zoom In":       "close-up shot, intimate framing",
    "Zoom Out":      "wide establishing shot, expansive view",
    "Pan Left":      "dynamic angle, action framing",
    "Pan Right":     "dynamic angle, sweeping view",
    "Tilt Up":       "low-angle shot, towering perspective",
    "Tilt Down":     "high-angle shot, overhead perspective",
    "Tracking Shot": "tracking shot, dynamic motion framing",
    "Slow Zoom":     "cinematic medium shot, gentle framing",
}

# ── Emotion lighting keywords ─────────────────────────────────────────────────

EMOTION_KEYWORDS: Dict[str, str] = {
    "tense":         "tense atmosphere, dramatic shadows, high contrast",
    "sad":           "somber mood, muted colors, soft diffuse lighting",
    "melancholic":   "melancholic atmosphere, cool tones, soft rain",
    "happy":         "warm bright atmosphere, cheerful lighting",
    "joyful":        "vibrant joyful atmosphere, warm golden light",
    "intense":       "intense dramatic atmosphere, dark moody lighting",
    "fearful":       "ominous dark atmosphere, cool blue shadows",
    "triumphant":    "triumphant atmosphere, bright dramatic lighting",
    "peaceful":      "peaceful serene atmosphere, soft natural light",
    "romantic":      "romantic mood, warm soft lighting, gentle glow",
    "mysterious":    "mysterious atmosphere, fog, dim dramatic lighting",
    "suspenseful":   "suspenseful tension, stark lighting, deep shadows",
    "rage":          "explosive intense atmosphere, harsh red-orange lighting",
}


class PromptBuilder:
    """
    Structured image prompt assembler.

    Assembles prompts in a deterministic order so the most important
    information (base description + characters) comes first.

    Usage
    -----
    builder = PromptBuilder(world_engine, director_profile, project_dir)
    prompt = builder.build(scene_dict)
    # → "Hero fights in the dark forest at night. Hero: tall, dark hair...
    #    moonlit, deep shadows. close-up shot. tense atmosphere.
    #    cinematic photography, dramatic lighting. masterpiece, 8k..."
    """

    def __init__(
        self,
        world_engine: Optional[Any] = None,
        director_profile: Optional[Dict[str, Any]] = None,
        project_dir: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        world_engine : WorldEngine, optional
            Provides character + location context.
        director_profile : dict, optional
            From director_profile.json — provides image_style and quality_suffix.
        project_dir : Path, optional
            Used to load character prompt files as fallback.
        """
        self.world = world_engine
        self.profile = director_profile or {}
        self.project_dir = Path(project_dir) if project_dir else None

    def build(self, scene: Dict[str, Any]) -> str:
        """
        Assemble the full image prompt for a scene.

        Assembly order:
          1. Base description (scene image_prompt)
          2. Character visual state
          3. Location atmosphere
          4. Time-of-day lighting
          5. Camera/shot keywords
          6. Emotion keywords
          7. Image style (from director profile)

        Returns
        -------
        str
            The assembled prompt (before optimization/validation).
        """
        parts: List[str] = []

        # 1. Base description
        base = scene.get("image_prompt", "").strip()
        if base:
            parts.append(base)

        # 2. Character visual state
        char_names = scene.get("characters", [])
        if char_names:
            char_ctx = self._get_character_context(char_names)
            if char_ctx:
                parts.append(char_ctx)

        # 3. Location atmosphere
        location = scene.get("location", "")
        if location:
            loc_ctx = self._get_location_context(location)
            if loc_ctx:
                parts.append(loc_ctx)

        # 4. Time-of-day lighting
        time_of_day = scene.get("time_of_day", "")
        if not time_of_day and self.world:
            time_of_day = self.world.current_time_of_day
        if time_of_day:
            lighting = self._get_time_lighting(time_of_day)
            if lighting:
                parts.append(lighting)

        # 5. Camera / shot type
        camera_motion = scene.get("camera_motion", "Static")
        cam_kw = CAMERA_KEYWORDS.get(camera_motion, "")
        if cam_kw:
            parts.append(cam_kw)

        # 6. Emotion keywords
        emotion = scene.get("emotion", "neutral")
        emotion_kw = EMOTION_KEYWORDS.get(emotion.lower().strip(), "")
        if emotion_kw:
            parts.append(emotion_kw)

        # 7. Image style from director profile
        image_style = self.profile.get("image_style", "")
        if image_style:
            parts.append(image_style)

        # Join with period-space separator, clean up
        prompt = ". ".join(p.rstrip(".").strip() for p in parts if p.strip())
        return prompt

    # ── Context helpers ───────────────────────────────────────────────────────

    def _get_character_context(self, names: List[str]) -> str:
        """Get character visual state from WorldEngine or fallback files."""
        if self.world:
            return self.world.get_character_context(names)

        # Fallback: read character prompt.txt files
        if self.project_dir:
            parts = []
            chars_dir = self.project_dir / "characters"
            for name in names:
                stem = re.sub(r"[^\w\-]", "_", name.strip()).lower()
                prompt_file = chars_dir / f"{stem}_prompt.txt"
                if prompt_file.exists():
                    txt = prompt_file.read_text(encoding="utf-8").strip()
                    if txt:
                        parts.append(f"{name}: {txt}")
            return "\n".join(parts)

        return ""

    def _get_location_context(self, location: str) -> str:
        """Get location atmosphere from WorldEngine."""
        if self.world:
            return self.world.get_location_context(location)
        return ""

    def _get_time_lighting(self, time_of_day: str) -> str:
        """Map time-of-day to lighting keywords."""
        # Check director profile first (style-specific overrides)
        profile_lighting = self.profile.get("time_of_day_lighting", {})
        if profile_lighting.get(time_of_day.lower()):
            return profile_lighting[time_of_day.lower()]

        return TIME_LIGHTING.get(time_of_day.lower().strip(), "")

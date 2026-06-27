"""
lib/agents/director_agent.py — Creative director for the pipeline.

Determines the visual style, pacing, narration tone, and camera approach
for the entire project. All downstream modules consume director_profile.json
rather than hardcoded values.

Workflow
--------
1. Load the style profile from styles/{style}.json
2. Optionally use QwenClient to analyze novel tone and refine direction
3. Write director_profile.json to the project directory

Every module that needs style info loads director_profile.json.
Changing style = change one config option, nothing else.

Output: projects/{name}/director_profile.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from lib.agents import BaseAgent

logger = logging.getLogger(__name__)

TONE_ANALYSIS_PROMPT = """
You are an experienced creative director analyzing a novel for video adaptation.

Story sample:
{text_sample}

Analyze the overall tone, pacing, and emotional intensity of this content.
Output a JSON object with these fields:
{{
  "tone": "<dark|light|balanced|intense|humorous|melancholic>",
  "pacing": "<slow|medium|fast|varied>",
  "emotion_intensity": "<low|medium|high>",
  "dominant_themes": ["<theme1>", "<theme2>"],
  "recommended_camera_style": "<static|dynamic|cinematic|slow_pan>",
  "recommended_transition_style": "<cut|dissolve|fast_cut>",
  "narration_style_override": "<dramatic|calm|intense|energetic|neutral>",
  "reasoning": "<1-2 sentence explanation>"
}}

Output ONLY the JSON object, no explanation.
""".strip()


class DirectorAgent(BaseAgent):
    """
    Creative director — loads style profile and optionally refines it
    using LLM analysis of the source text.

    The director_profile.json produced by this agent is consumed by:
    - PromptBuilder     (image_style, quality_suffix)
    - MotionPlanner     (camera_style, transition_style)
    - MetadataGenerator (style name for description)
    - ThumbnailGenerator (color palette)
    """

    def __init__(
        self,
        project_name: str,
        style: str = "cinematic",
        use_llm_analysis: bool = False,
        styles_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        project_name : str
        style : str
            Style profile ID (matches filename in styles/ dir)
        use_llm_analysis : bool
            If True, use Qwen to refine the style profile based on text tone.
            Costs ~1 extra Qwen call per project (runs once, cached).
        styles_dir : str, optional
            Override path to styles/ directory
        """
        super().__init__(project_name, **kwargs)
        self.style = style
        self.use_llm_analysis = use_llm_analysis

        # Resolve styles directory
        if styles_dir:
            self.styles_dir = Path(styles_dir)
        else:
            # Look for styles/ relative to project root or repo root
            repo_root = Path(__file__).parent.parent.parent
            self.styles_dir = repo_root / "styles"
            if not self.styles_dir.exists():
                self.styles_dir = Path("styles")

    # ── Style loading ─────────────────────────────────────────────────────────

    def _load_style_profile(self) -> Dict[str, Any]:
        """Load style profile from styles/{style}.json."""
        style_file = self.styles_dir / f"{self.style}.json"

        if style_file.exists():
            with open(style_file, encoding="utf-8") as f:
                profile = json.load(f)
            self.log(f"Loaded style profile: {style_file}")
            return profile

        # Style not found — use cinematic as fallback
        self.log(f"Style '{self.style}' not found at {style_file}, using cinematic")
        fallback = self.styles_dir / "cinematic.json"
        if fallback.exists():
            with open(fallback, encoding="utf-8") as f:
                return json.load(f)

        # Ultimate fallback: built-in cinematic defaults
        return self._default_cinematic_profile()

    @staticmethod
    def _default_cinematic_profile() -> Dict[str, Any]:
        """Built-in cinematic defaults if no style files exist."""
        return {
            "id": "cinematic",
            "name": "Cinematic",
            "image_style": "cinematic photography, dramatic lighting",
            "quality_suffix": "masterpiece, best quality, highly detailed, 8k, sharp focus",
            "camera_style": "cinematic",
            "narration_style": "dramatic",
            "transition_style": "dissolve",
            "emotion_intensity": "medium",
            "target_audience": "general",
            "preferred_model": "flux_schnell",
            "color_palette": "desaturated",
            "aspect_ratio": "16:9",
            "time_of_day_lighting": {
                "dawn": "golden hour, warm orange light",
                "day": "bright natural light, soft shadows",
                "dusk": "purple twilight, dramatic sky",
                "night": "moonlit, deep shadows",
            },
        }

    # ── LLM tone analysis ─────────────────────────────────────────────────────

    def _analyze_tone(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use QwenClient to analyze the story tone and refine the style profile.
        Runs only when use_llm_analysis=True (costs ~1 extra inference).
        """
        # Load story segments for analysis
        try:
            seg_path = self.get_path("drafts", "episode_1", "step1_segments.json")
            if not seg_path.exists():
                self.log("No segments file for tone analysis — skipping LLM refinement")
                return profile

            with open(seg_path, encoding="utf-8") as f:
                segments = json.load(f)

            # Sample first 20 segments for tone analysis
            sample_text = "\n".join(
                s.get("novel_text", "")[:200] for s in segments[:20]
            )

            prompt = TONE_ANALYSIS_PROMPT.format(text_sample=sample_text)
            tone_data = self.qwen.generate_json(prompt, temperature=0.1, thinking=False)

            if not isinstance(tone_data, dict):
                return profile

            # Selectively apply LLM suggestions to the loaded profile
            if tone_data.get("emotion_intensity"):
                profile["emotion_intensity"] = tone_data["emotion_intensity"]
            if tone_data.get("recommended_camera_style"):
                profile["camera_style"] = tone_data["recommended_camera_style"]
            if tone_data.get("recommended_transition_style"):
                profile["transition_style"] = tone_data["recommended_transition_style"]
            if tone_data.get("narration_style_override"):
                profile["narration_style"] = tone_data["narration_style_override"]

            profile["tone_analysis"] = {
                "tone": tone_data.get("tone"),
                "pacing": tone_data.get("pacing"),
                "dominant_themes": tone_data.get("dominant_themes", []),
                "reasoning": tone_data.get("reasoning", ""),
            }
            self.log(f"Tone analysis: {tone_data.get('tone', 'unknown')} / {tone_data.get('pacing', 'unknown')}")

        except Exception as e:
            self.log(f"Tone analysis failed ({e}) — using base style profile")

        return profile

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, overwrite: bool = False) -> Dict[str, Any]:
        """
        Generate and save director_profile.json.

        Returns
        -------
        dict
            The director profile
        """
        profile_path = self.project_dir / "director_profile.json"

        # Resume: use existing profile if not overwriting
        if profile_path.exists() and not overwrite:
            self.log("Director profile already exists — skipping")
            with open(profile_path, encoding="utf-8") as f:
                return json.load(f)

        self.log(f"Creating director profile (style='{self.style}', llm={self.use_llm_analysis})")

        # 1. Load base style profile
        profile = self._load_style_profile()

        # 2. Optionally refine with LLM tone analysis
        if self.use_llm_analysis:
            profile = self._analyze_tone(profile)

        # 3. Add runtime metadata
        from datetime import datetime
        profile["_meta"] = {
            "created_at": datetime.now().isoformat(),
            "style_requested": self.style,
            "llm_analyzed": self.use_llm_analysis,
            "project": self.project_name,
        }

        # 4. Save
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

        self.log(f"Director profile saved: {profile_path}")
        return profile

    @staticmethod
    def load_profile(project_dir: Path) -> Dict[str, Any]:
        """
        Load director profile from project directory.

        Called by PromptBuilder, MotionPlanner, and other downstream modules.
        Returns empty dict if no profile exists (graceful fallback).
        """
        profile_path = Path(project_dir) / "director_profile.json"
        if profile_path.exists():
            with open(profile_path, encoding="utf-8") as f:
                return json.load(f)
        logger.warning(f"No director_profile.json found in {project_dir} — using defaults")
        return DirectorAgent._default_cinematic_profile()

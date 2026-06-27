"""
lib/motion_planner.py — Intelligent camera motion selector.

Uses a rule engine (always) + optional LLM override (when MOTION_PLANNER_LLM_HINTS=True)
to select camera motion, speed, and transition style for each scene.

Rule Engine
-----------
Decision table: scene_type × emotion × is_key_scene → motion + speed

LLM Override (optional)
-----------------------
When enabled, the rule-engine suggestion is sent to QwenClient with the
scene summary. The LLM can override for dramatic effect. LLM decisions
are cached so each scene type is only queried once.

Output
------
MotionPlan dataclass per scene. Consumed by StoryboardAgent and MotionEngine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Motion decision table ─────────────────────────────────────────────────────
# Format: (scene_type, emotion_hint) → (camera_motion, speed, duration_hint_s)
#
# scene_type is from the StoryboardAgent controlled vocabulary.
# emotion_hint is a simplified grouping of the scene's emotion field.
# If no exact match, falls back to the scene_type-only match.
# If still no match, uses DEFAULT_MOTION.

MOTION_TABLE: Dict[Tuple[str, str], Tuple[str, str, float]] = {
    # Battle / Action
    ("battle", "intense"):     ("Pan Left",   "fast",   3.0),
    ("battle", "calm"):        ("Zoom In",    "medium", 4.0),
    ("battle", "any"):         ("Pan Right",  "fast",   3.0),
    ("action", "intense"):     ("Pan Left",   "fast",   3.0),
    ("action", "any"):         ("Pan Right",  "medium", 4.0),

    # Dialogue
    ("dialogue", "tense"):     ("Zoom In",    "slow",   5.0),
    ("dialogue", "sad"):       ("Zoom In",    "slow",   6.0),
    ("dialogue", "happy"):     ("Static",     "slow",   5.0),
    ("dialogue", "any"):       ("Static",     "slow",   5.0),

    # Emotional
    ("emotional", "sad"):      ("Zoom In",    "slow",   6.0),
    ("emotional", "happy"):    ("Zoom Out",   "slow",   5.0),
    ("emotional", "intense"):  ("Zoom In",    "slow",   6.0),
    ("emotional", "any"):      ("Zoom In",    "slow",   5.0),

    # Travel / Exterior
    ("travel", "any"):         ("Pan Right",  "medium", 4.0),
    ("city", "any"):           ("Pan Right",  "medium", 4.5),
    ("forest", "any"):         ("Pan Left",   "slow",   5.0),
    ("exterior", "any"):       ("Tilt Up",    "slow",   5.0),

    # Interior / Scenes
    ("interior", "any"):       ("Static",     "slow",   4.0),
    ("tavern", "any"):         ("Pan Left",   "medium", 4.0),
    ("castle", "any"):         ("Tilt Up",    "slow",   5.5),
    ("classroom", "any"):      ("Static",     "slow",   4.5),

    # Ceremony / Exposition
    ("ceremony", "any"):       ("Static",     "slow",   6.0),
    ("exposition", "any"):     ("Slow Zoom",  "slow",   5.0),

    # Training
    ("training", "intense"):   ("Pan Left",   "medium", 4.0),
    ("training", "any"):       ("Pan Left",   "medium", 4.5),
}

# Scene types that never appear in the table fall through to default
DEFAULT_MOTION = ("Static", "medium", 4.0)

# Key scene override: always use slow cinematic motion for maximum impact
KEY_SCENE_OVERRIDE = ("Zoom In", "slow", 6.0)

# Filler scene override: always use static for maximum speed
FILLER_SCENE_OVERRIDE = ("Static", "medium", 3.5)

# Emotion grouping: maps detailed emotions to simplified bucket
EMOTION_GROUPS: Dict[str, str] = {
    "tense": "tense", "anxious": "tense", "fearful": "tense", "suspenseful": "tense",
    "sad": "sad", "sorrowful": "sad", "melancholic": "sad", "grief": "sad",
    "happy": "happy", "joyful": "happy", "triumphant": "happy", "hopeful": "happy",
    "intense": "intense", "dramatic": "intense", "rage": "intense", "fury": "intense",
    "calm": "calm", "peaceful": "calm", "neutral": "calm",
}


@dataclass
class MotionPlan:
    """Camera motion plan for a single scene."""
    camera_motion: str    # "Static", "Pan Left", etc.
    speed: str            # "slow", "medium", "fast"
    transition: str       # "crossfade", "cut", "dissolve"
    duration_hint: float  # suggested clip duration in seconds
    source: str           # "key_scene" | "filler_scene" | "rule" | "llm"

    def to_dict(self) -> Dict:
        return asdict(self)


class MotionPlanner:
    """
    Intelligent camera motion selector.

    Combines a rule engine (always fast, deterministic) with optional
    LLM overrides (better quality, costs tokens, results are cached).

    Usage
    -----
    planner = MotionPlanner(director_profile, key_scenes=["scene_001"])
    plan = planner.plan(scene_dict)
    scene["camera_motion"] = plan.camera_motion
    """

    def __init__(
        self,
        director_profile: Optional[Dict[str, Any]] = None,
        key_scenes: Optional[List[str]] = None,
        filler_scenes: Optional[List[str]] = None,
        llm_hints_enabled: bool = False,
        llm_cache: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        director_profile : dict, optional
            From director_profile.json. Adjusts camera_style.
        key_scenes : list, optional
            Scene IDs that are key/climactic — get slow cinematic treatment.
        filler_scenes : list, optional
            Scene IDs that are filler — get static/fast treatment.
        llm_hints_enabled : bool
            If True, allow QwenClient to override rule decisions.
        llm_cache : dict, optional
            Persistent cache for LLM decisions (avoids re-querying).
        """
        self.profile = director_profile or {}
        self.key_scenes: set = set(key_scenes or [])
        self.filler_scenes: set = set(filler_scenes or [])
        self.llm_hints_enabled = llm_hints_enabled
        self._llm_cache: Dict[str, MotionPlan] = {}
        if llm_cache:
            for k, v in llm_cache.items():
                self._llm_cache[k] = MotionPlan(**v)
        self._qwen: Any = None

    def _get_transition(self, speed: str) -> str:
        """Map speed to transition style."""
        style = self.profile.get("transition_style", "dissolve")
        if style == "fast_cut":
            return "cut"
        if speed == "fast":
            return "cut"
        if speed == "slow":
            return "dissolve"
        return "crossfade"

    def _group_emotion(self, emotion: str) -> str:
        """Map detailed emotion → simplified bucket."""
        return EMOTION_GROUPS.get(emotion.lower().strip(), "any")

    def _rule_lookup(
        self,
        scene_type: str,
        emotion: str,
    ) -> Tuple[str, str, float]:
        """Look up motion from the rule table."""
        grouped = self._group_emotion(emotion)

        # Exact match
        if (scene_type, grouped) in MOTION_TABLE:
            return MOTION_TABLE[(scene_type, grouped)]

        # Scene type + any
        if (scene_type, "any") in MOTION_TABLE:
            return MOTION_TABLE[(scene_type, "any")]

        return DEFAULT_MOTION

    def plan(self, scene: Dict[str, Any]) -> MotionPlan:
        """
        Produce a MotionPlan for a scene.

        Priority:
          1. Key scene override → always slow cinematic
          2. Filler scene override → always static
          3. Rule engine lookup
          4. LLM override (if enabled and confident)
        """
        scene_id = scene.get("segment_id", "")
        scene_type = scene.get("scene_type", "exposition")
        emotion = scene.get("emotion", "neutral")

        # 1. Key scene override
        if scene_id and scene_id in self.key_scenes:
            motion, speed, duration = KEY_SCENE_OVERRIDE
            return MotionPlan(
                camera_motion=motion,
                speed=speed,
                transition=self._get_transition(speed),
                duration_hint=duration,
                source="key_scene",
            )

        # 2. Filler scene override
        if scene_id and scene_id in self.filler_scenes:
            motion, speed, duration = FILLER_SCENE_OVERRIDE
            return MotionPlan(
                camera_motion=motion,
                speed=speed,
                transition=self._get_transition(speed),
                duration_hint=duration,
                source="filler_scene",
            )

        # 3. Rule engine
        motion, speed, duration = self._rule_lookup(scene_type, emotion)
        plan = MotionPlan(
            camera_motion=motion,
            speed=speed,
            transition=self._get_transition(speed),
            duration_hint=duration,
            source="rule",
        )

        # 4. Optional LLM override
        if self.llm_hints_enabled:
            llm_plan = self._llm_override(scene, plan)
            if llm_plan:
                return llm_plan

        return plan

    def _llm_override(
        self,
        scene: Dict[str, Any],
        rule_plan: MotionPlan,
    ) -> Optional[MotionPlan]:
        """
        Ask QwenClient if the rule-based decision should be overridden.
        Results are cached by (scene_type, emotion) so each combo is only queried once.
        """
        cache_key = f"{scene.get('scene_type', '')}_{scene.get('emotion', '')}"
        if cache_key in self._llm_cache:
            cached = self._llm_cache[cache_key]
            # Return cached only if confidence is high
            if cached.source == "llm":
                return MotionPlan(
                    camera_motion=cached.camera_motion,
                    speed=cached.speed,
                    transition=cached.transition,
                    duration_hint=cached.duration_hint,
                    source="llm_cached",
                )

        try:
            if self._qwen is None:
                from lib.qwen_client import get_qwen_client
                self._qwen = get_qwen_client()

            prompt = (
                f"Scene type: {scene.get('scene_type', 'exposition')}\n"
                f"Emotion: {scene.get('emotion', 'neutral')}\n"
                f"Summary: {scene.get('novel_text', '')[:150]}\n"
                f"Rule-based suggestion: {rule_plan.camera_motion} ({rule_plan.speed})\n\n"
                "Should the camera motion be overridden for dramatic effect?\n"
                'Output JSON: {"override": true/false, "camera_motion": "...", '
                '"speed": "slow/medium/fast", "duration_hint": 4.0, "confidence": 0.0-1.0}'
            )

            result = self._qwen.generate_json(prompt, temperature=0.1, thinking=False)

            if (
                isinstance(result, dict)
                and result.get("override") is True
                and result.get("confidence", 0) >= 0.8
            ):
                motion = result.get("camera_motion", rule_plan.camera_motion)
                speed = result.get("speed", rule_plan.speed)
                llm_plan = MotionPlan(
                    camera_motion=motion,
                    speed=speed,
                    transition=self._get_transition(speed),
                    duration_hint=float(result.get("duration_hint", rule_plan.duration_hint)),
                    source="llm",
                )
                self._llm_cache[cache_key] = llm_plan
                return llm_plan

        except Exception as e:
            logger.debug(f"[MotionPlanner] LLM override failed: {e}")

        return None

    def export_llm_cache(self) -> Dict:
        """Export LLM decision cache for persistence."""
        return {k: v.to_dict() for k, v in self._llm_cache.items()}

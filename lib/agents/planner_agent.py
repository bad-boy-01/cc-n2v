"""
lib/agents/planner_agent.py — Episode planner and scene budget allocator.

Sits between StoryAgent and DirectorAgent in the pipeline.
Uses QwenClient to analyze the story structure and produce an episode plan:
  - Episode splits (by word count or narrative arcs)
  - Runtime estimates
  - Key vs. filler scene identification
  - GPU budget per episode

Output: drafts/episode_N/episode_plan.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.agents import BaseAgent

logger = logging.getLogger(__name__)

# Key scene markers in story text (used for heuristic detection)
KEY_SCENE_KEYWORDS = {
    "confrontation", "battle", "duel", "reveal", "betrayal",
    "sacrifice", "death", "awakening", "transformation", "climax",
    "finally", "at last", "suddenly", "screamed", "explosion",
    "defeated", "victory", "surrendered", "fled", "kissed",
}

FILLER_SCENE_KEYWORDS = {
    "meanwhile", "later", "walked to", "sat down", "ate",
    "traveled", "arrived at", "prepared", "gathered",
}

PLANNER_PROMPT = """
You are an experienced narrative director planning a video adaptation of a novel.

Story segments (total: {total_segments}):
{segment_summaries}

Create an episode plan that:
1. Groups segments into episodes of roughly {target_episode_length} segments each
2. Identifies KEY scenes (climactic, emotional peaks) vs FILLER scenes (transitions, travel)
3. Estimates runtime per episode (assume {seconds_per_segment}s per segment)

Output a JSON object with this structure:
{{
  "total_episodes": <int>,
  "episode_plans": [
    {{
      "episode": 1,
      "title": "<2-5 word title>",
      "segment_range": [<start_idx>, <end_idx>],
      "estimated_segments": <int>,
      "estimated_runtime_minutes": <float>,
      "key_scene_ids": ["<segment_id>", ...],
      "filler_scene_ids": ["<segment_id>", ...],
      "pacing": "<slow_build|steady|fast|climactic>",
      "arc": "<introduction|rising_action|climax|falling_action|resolution>"
    }}
  ]
}}

CRITICAL: Output ONLY the JSON object, no explanation.
""".strip()


class PlannerAgent(BaseAgent):
    """
    Episode planner — splits story into episodes and classifies scenes.

    Key scenes get:
    - Detailed FLUX prompts
    - Slow cinematic camera motion
    - Priority in quality checking

    Filler scenes get:
    - Simpler prompts (faster generation)
    - SDXL routing (lower VRAM)
    - Static camera motion
    """

    def __init__(
        self,
        project_name: str,
        episode: int = 1,
        target_segments_per_episode: int = 45,
        **kwargs,
    ):
        super().__init__(project_name, **kwargs)
        self.episode = episode
        self.target_segments_per_episode = target_segments_per_episode

    def _load_segments(self) -> List[Dict]:
        """Load step1_segments.json from StoryAgent output."""
        path = self.get_path("drafts", f"episode_{self.episode}", "step1_segments.json")
        if not path.exists():
            raise FileNotFoundError(
                f"Run StoryAgent first. Missing: {path}"
            )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _heuristic_classify(
        self, segments: List[Dict]
    ) -> tuple[List[str], List[str]]:
        """
        Fast heuristic classification of key vs. filler scenes.
        Used when LLM output is unavailable or as a pre-pass before LLM.
        """
        key_ids = []
        filler_ids = []
        for seg in segments:
            text = seg.get("novel_text", "").lower()
            seg_id = seg.get("segment_id", "")
            if any(kw in text for kw in KEY_SCENE_KEYWORDS):
                key_ids.append(seg_id)
            elif any(kw in text for kw in FILLER_SCENE_KEYWORDS):
                filler_ids.append(seg_id)
        return key_ids, filler_ids

    def _build_summaries(self, segments: List[Dict], max_segments: int = 100) -> str:
        """Build compact segment summary for LLM (truncate for token budget)."""
        sample = segments[:max_segments]
        lines = []
        for i, seg in enumerate(sample):
            text = seg.get("novel_text", "")[:120].replace("\n", " ")
            lines.append(f"[{i:03d}] {seg.get('segment_id', '')}: {text}…")
        if len(segments) > max_segments:
            lines.append(f"... ({len(segments) - max_segments} more segments not shown)")
        return "\n".join(lines)

    def run(self, overwrite: bool = False) -> Dict[str, Any]:
        """
        Generate episode plan.

        Returns
        -------
        dict
            plan dict with total_episodes, episode_plans, key_scenes, filler_scenes
        """
        plan_path = self.get_path("drafts", f"episode_{self.episode}", "episode_plan.json")

        # Resume: load existing plan if not overwriting
        if plan_path.exists() and not overwrite:
            self.log("Episode plan already exists — skipping (use overwrite=True to regenerate)")
            with open(plan_path, encoding="utf-8") as f:
                return json.load(f)

        segments = self._load_segments()
        total = len(segments)
        self.log(f"Planning {total} segments into episodes of ~{self.target_segments_per_episode}")

        # Fast heuristic classification (always runs)
        heuristic_keys, heuristic_fillers = self._heuristic_classify(segments)

        # Try LLM-based planning for better episode splits
        plan_data: Dict[str, Any] = {}
        try:
            prompt = PLANNER_PROMPT.format(
                total_segments=total,
                segment_summaries=self._build_summaries(segments),
                target_episode_length=self.target_segments_per_episode,
                seconds_per_segment=4,
            )
            plan_data = self.qwen.generate_json(prompt, temperature=0.1, thinking=False)
            self.log(f"LLM episode plan: {plan_data.get('total_episodes', '?')} episodes")
        except Exception as e:
            self.log(f"LLM planning failed ({e}), using heuristic episode split")

        # Fall back to word-count-based episode split
        if not isinstance(plan_data, dict) or "episode_plans" not in plan_data:
            plan_data = self._heuristic_plan(segments, heuristic_keys, heuristic_fillers)

        # Merge heuristic key/filler into LLM plan (in case LLM missed some)
        all_key_ids: set = set(heuristic_keys)
        all_filler_ids: set = set(heuristic_fillers)

        for ep in plan_data.get("episode_plans", []):
            all_key_ids.update(ep.get("key_scene_ids", []))
            all_filler_ids.update(ep.get("filler_scene_ids", []))
            # Ensure no overlap
            ep["key_scene_ids"] = sorted(set(ep.get("key_scene_ids", [])) | all_key_ids)
            ep["filler_scene_ids"] = sorted(
                (set(ep.get("filler_scene_ids", [])) | all_filler_ids)
                - set(ep["key_scene_ids"])  # key always wins over filler
            )

        plan_data["total_segments"] = total
        plan_data["heuristic_key_scenes"] = sorted(all_key_ids)
        plan_data["heuristic_filler_scenes"] = sorted(
            all_filler_ids - all_key_ids
        )

        # Save
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, ensure_ascii=False, indent=2)

        self.log(f"Episode plan saved: {plan_path}")
        return plan_data

    def _heuristic_plan(
        self,
        segments: List[Dict],
        key_ids: List[str],
        filler_ids: List[str],
    ) -> Dict[str, Any]:
        """Fallback: simple word-count-based episode split."""
        n_episodes = max(1, -(-len(segments) // self.target_segments_per_episode))
        episode_plans = []

        for ep_idx in range(n_episodes):
            start = ep_idx * self.target_segments_per_episode
            end = min(start + self.target_segments_per_episode, len(segments))
            ep_segs = segments[start:end]
            ep_seg_ids = {s.get("segment_id", "") for s in ep_segs}

            episode_plans.append({
                "episode": ep_idx + 1,
                "title": f"Episode {ep_idx + 1}",
                "segment_range": [start, end - 1],
                "estimated_segments": len(ep_segs),
                "estimated_runtime_minutes": round(len(ep_segs) * 4 / 60, 1),
                "key_scene_ids": [s for s in key_ids if s in ep_seg_ids],
                "filler_scene_ids": [s for s in filler_ids if s in ep_seg_ids],
                "pacing": "steady",
                "arc": (
                    "introduction" if ep_idx == 0
                    else "resolution" if ep_idx == n_episodes - 1
                    else "rising_action"
                ),
            })

        return {
            "total_episodes": n_episodes,
            "episode_plans": episode_plans,
        }

    def get_key_scenes(self) -> List[str]:
        """Return list of key scene IDs from saved plan."""
        plan_path = self.get_path("drafts", f"episode_{self.episode}", "episode_plan.json")
        if not plan_path.exists():
            return []
        with open(plan_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("heuristic_key_scenes", [])

    def get_filler_scenes(self) -> List[str]:
        """Return list of filler scene IDs from saved plan."""
        plan_path = self.get_path("drafts", f"episode_{self.episode}", "episode_plan.json")
        if not plan_path.exists():
            return []
        with open(plan_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("heuristic_filler_scenes", [])

"""
lib/agents/storyboard_agent.py — Storyboard JSON generation with batching.

Ports: .claude/skills/generate-script/SKILL.md

Key design decisions:
- MAX_SCENES_PER_BATCH = 25  (prevents Qwen context overflow, enables resume)
- Saves checkpoint after EVERY batch
- Scene cache key uses {location, sorted(character_ids), scene_type} — NOT raw action text
- scene_type is from a controlled vocabulary for high cache-hit rate

Scene type vocabulary:
    dialogue, battle, travel, exposition, emotional,
    training, city, castle, tavern, classroom, forest,
    interior, exterior, action, ceremony
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.agents import BaseAgent
from lib.agents.character_agent import CharacterAgent
from lib.agents.clue_agent import ClueAgent
from lib.world_engine import WorldEngine
from lib.prompt_builder import PromptBuilder
from lib.prompt_optimizer import PromptOptimizer
from lib.prompt_validator import PromptValidator

MAX_SCENES_PER_BATCH = 25

# Controlled vocabulary for scene_type (drives cache hit rate)
SCENE_TYPES = {
    "dialogue", "battle", "travel", "exposition", "emotional",
    "training", "city", "castle", "tavern", "classroom", "forest",
    "interior", "exterior", "action", "ceremony",
}

STORYBOARD_BATCH_PROMPT = """
You are a visual storyboard director converting novel segments into scene descriptions.

Project style: {style}
Known characters: {character_summary}
Known locations: {location_summary}

Convert each of the following {count} story segments into a storyboard scene.
Output a JSON array with exactly {count} objects, one per segment, in order.

Each object must have:
{{
  "segment_id": "<copy from input>",
  "title": "<2-5 word scene title>",
  "location": "<specific location name>",
  "characters": ["<list of character names present>"],
  "scene_type": "<one of: {scene_types}>",
  "emotion": "<dominant emotional tone>",
  "image_prompt": "<detailed visual description for image generation, 2-3 sentences>",
  "camera_motion": "<one of: Static, Pan Left, Pan Right, Zoom In, Zoom Out, Tilt Up, Tilt Down>"
}}

Segments to process:
{segments_json}

CRITICAL:
- image_prompt must be self-contained and visually specific
- scene_type must be exactly one word from the allowed list
- characters list must use exact names from the known characters list
- Output ONLY the JSON array, no explanation
""".strip()


def _normalize_scene_type(raw: str) -> str:
    """Map LLM output to controlled vocabulary, defaulting to 'exposition'."""
    raw = raw.lower().strip()
    if raw in SCENE_TYPES:
        return raw
    # Fuzzy match
    for st in SCENE_TYPES:
        if st in raw or raw in st:
            return st
    return "exposition"


class StoryboardAgent(BaseAgent):
    """
    Generates storyboard JSON from segments in MAX_SCENES_PER_BATCH=25 chunks.
    Saves a checkpoint after each batch so the pipeline can resume.
    """

    def __init__(
        self,
        project_name: str,
        episode: int = 1,
        max_scenes_per_batch: int = MAX_SCENES_PER_BATCH,
        **kwargs,
    ):
        super().__init__(project_name, **kwargs)
        self.episode = episode
        self.max_scenes_per_batch = max_scenes_per_batch
        self._char_agent = CharacterAgent(project_name, episode, **kwargs)
        self._clue_agent = ClueAgent(project_name, episode, **kwargs)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_segments(self) -> List[Dict]:
        path = self.get_path("drafts", f"episode_{self.episode}", "step1_segments.json")
        if not path.exists():
            raise FileNotFoundError(f"Run StoryAgent first. Missing: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _load_project_style(self) -> str:
        try:
            project = self.pm.load_project(self.project_name)
            overview = project.get("overview", {})
            return f"{overview.get('genre', '')} — {overview.get('world_setting', '')}"
        except Exception:
            return "cinematic story"

    def _build_character_summary(self) -> str:
        chars_dir = self.get_path("characters")
        if not chars_dir.exists():
            return "No characters defined yet."
        summaries = []
        for prompt_file in sorted(chars_dir.glob("*_prompt.txt")):
            name = prompt_file.stem.replace("_prompt", "").replace("_", " ").title()
            summaries.append(f"- {name}: {prompt_file.read_text(encoding='utf-8').strip()}")
        return "\n".join(summaries) if summaries else "No characters defined yet."

    def _build_location_summary(self) -> str:
        clues_dir = self.get_path("clues")
        if not clues_dir.exists():
            return "No locations defined yet."
        summaries = []
        for jf in sorted(clues_dir.glob("*.json")):
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("type") == "location":
                summaries.append(f"- {data.get('name', jf.stem)}: {data.get('visual_description', '')}")
        return "\n".join(summaries) if summaries else "No locations defined yet."

    def _checkpoint_path(self) -> Path:
        return self.get_path("drafts", f"episode_{self.episode}", "storyboard_checkpoint.json")

    def _load_checkpoint(self) -> Dict[str, Any]:
        p = self._checkpoint_path()
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return {"completed_batches": [], "scenes": {}}

    def _save_checkpoint(self, state: Dict[str, Any]) -> None:
        p = self._checkpoint_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _process_batch(
        self,
        batch: List[Dict],
        style: str,
        char_summary: str,
        loc_summary: str,
        director_profile: Dict,
        world: WorldEngine,
    ) -> List[Dict]:
        """Send one batch to Qwen2.5 and return list of scene dicts."""
        # Strip fields not needed by LLM
        compact = [
            {
                "segment_id": s["segment_id"],
                "novel_text": s["novel_text"],
                "has_dialogue": s.get("has_dialogue", False),
                "segment_break": s.get("segment_break", False),
            }
            for s in batch
        ]

        prompt = STORYBOARD_BATCH_PROMPT.format(
            style=style,
            character_summary=char_summary,
            location_summary=loc_summary,
            count=len(batch),
            scene_types=", ".join(sorted(SCENE_TYPES)),
            segments_json=json.dumps(compact, ensure_ascii=False, indent=2),
        )

        raw = self.qwen.generate_json(prompt, temperature=0.2, thinking=False)

        # raw should be a list
        if isinstance(raw, dict):
            # Some models wrap in {"scenes": [...]}
            for v in raw.values():
                if isinstance(v, list):
                    raw = v
                    break
            else:
                raw = []

        if not isinstance(raw, list):
            self.log(f"WARNING: Unexpected LLM output type {type(raw)}, using empty list")
            raw = []

        # Setup prompt pipeline
        builder = PromptBuilder(world, director_profile, self.project_dir)
        optimizer = PromptOptimizer(director_profile)
        validator = PromptValidator(strict=False)

        # Merge LLM output back onto original segments
        results = []
        for i, seg in enumerate(batch):
            llm_data = raw[i] if i < len(raw) else {}
            if not isinstance(llm_data, dict):
                llm_data = {}
            scene = {
                # Preserve original segment fields
                **seg,
                # Overwrite with LLM-generated storyboard fields
                "title": llm_data.get("title", seg.get("segment_id", "")),
                "location": llm_data.get("location", "unknown"),
                "characters": llm_data.get("characters", []),
                "scene_type": _normalize_scene_type(llm_data.get("scene_type", "exposition")),
                "emotion": llm_data.get("emotion", "neutral"),
                "image_prompt": llm_data.get("image_prompt", seg.get("novel_text", "")),
                "camera_motion": llm_data.get("camera_motion", "Static"),
            }

            # Build rich prompt
            built = builder.build(scene)
            
            # Optimize prompt
            optimized = optimizer.optimize(built)
            
            # Validate prompt
            val_result = validator.validate(optimized, scene=scene)
            
            # Assign fixed prompt
            scene["image_prompt"] = val_result.fixed_prompt
            if val_result.warnings:
                scene["prompt_warnings"] = val_result.warnings

            results.append(scene)
            world.update_from_scene(scene)

        return results

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, overwrite: bool = False) -> Dict[str, Any]:
        """
        Generate storyboard for the episode in batches of MAX_SCENES_PER_BATCH.

        Saves a checkpoint after each batch.
        Resumes from checkpoint if overwrite=False (default).

        Returns
        -------
        dict: output_path, total_scenes, batches_processed
        """
        scripts_dir = self.ensure_dir("scripts")
        output_path = scripts_dir / f"episode_{self.episode}.json"

        segments = self._load_segments()
        total = len(segments)
        self.log(f"Generating storyboard for {total} segments in batches of {self.max_scenes_per_batch}")

        # Load checkpoint
        state = self._load_checkpoint() if not overwrite else {"completed_batches": [], "scenes": {}}
        completed_ids = set(state.get("completed_batches", []))
        scene_store: Dict[str, Dict] = state.get("scenes", {})

        # Split into batches
        batches = [
            segments[i: i + self.max_scenes_per_batch]
            for i in range(0, total, self.max_scenes_per_batch)
        ]

        style = self._load_project_style()
        char_summary = self._build_character_summary()
        loc_summary = self._build_location_summary()
        
        # Load DirectorProfile and WorldEngine
        profile_path = self.project_dir / "director_profile.json"
        director_profile = {}
        if profile_path.exists():
            with open(profile_path, encoding="utf-8") as f:
                director_profile = json.load(f)
                
        world = WorldEngine(self.project_name, str(self.project_dir.parent))
        world.load()

        batches_processed = 0
        for batch_idx, batch in enumerate(batches):
            batch_id = f"batch_{batch_idx:03d}"
            if batch_id in completed_ids:
                self.log(f"  Skipping completed {batch_id} ({len(batch)} scenes)")
                continue

            self.log(f"  Processing {batch_id} ({len(batch)} scenes) …")
            try:
                scenes = self._process_batch(
                    batch, style, char_summary, loc_summary, 
                    director_profile, world
                )
                for scene in scenes:
                    scene_store[scene["segment_id"]] = scene

                state["completed_batches"].append(batch_id)
                state["scenes"] = scene_store
                self._save_checkpoint(state)
                world.save_if_dirty()
                batches_processed += 1
                self.log(f"  ✅ {batch_id} complete, checkpoint saved")

            except Exception as e:
                self.log(f"  ❌ {batch_id} failed: {e} — skipping batch, will retry on next run")
                continue

        # Write final episode script (preserving NarrationEpisodeScript structure)
        all_scenes = [scene_store[s["segment_id"]] for s in segments if s["segment_id"] in scene_store]

        episode_script = {
            "episode": self.episode,
            "title": f"Episode {self.episode}",
            "content_mode": "narration",
            "duration_seconds": sum(s.get("duration_seconds", 4) for s in all_scenes),
            "summary": "",
            "novel": {
                "title": self.project_name,
                "chapter": f"Episode {self.episode}",
                "source_file": "",
            },
            "characters_in_episode": sorted({
                c for s in all_scenes for c in s.get("characters", [])
            }),
            "clues_in_episode": sorted({
                s.get("location", "") for s in all_scenes if s.get("location")
            }),
            "segments": all_scenes,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(episode_script, f, ensure_ascii=False, indent=2)

        self.log(f"Storyboard written to {output_path} ({len(all_scenes)}/{total} scenes)")

        return {
            "output_path": str(output_path),
            "total_scenes": len(all_scenes),
            "total_segments": total,
            "batches_processed": batches_processed,
            "complete": len(all_scenes) == total,
        }

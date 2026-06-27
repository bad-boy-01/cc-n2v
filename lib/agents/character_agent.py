"""
lib/agents/character_agent.py — Character extraction + reference portrait generation.

Ports: .claude/agents/novel-to-narration-script.md Step 2 (character section)

Workflow
--------
1. Read segments from step1_segments.json
2. Use Qwen2.5 to extract character visual descriptions
3. Write hero.json + hero_prompt.txt per character
4. Generate reference portrait via image_generator (deferred — called later in Stage 2)
5. Store to projects/{name}/characters/

Character consistency hierarchy (ALWAYS in this order):
  1. Prompt Template  (hero_prompt.txt)   ← PRIMARY
  2. Character JSON   (hero.json)
  3. Reference Image  (hero_ref.png)
  4. IP-Adapter       (optional, if VRAM allows)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.agents import BaseAgent
from lib.project_manager import ProjectManager
from lib.world_engine import WorldEngine

CHARACTER_SCHEMA = {
    "name": "character name",
    "gender": "male/female/unknown",
    "age": "approximate age or range",
    "hair": "hair color, length, style",
    "eyes": "eye color, shape, notable features",
    "clothing": "outfit description with colors and style",
    "body_type": "height and build description",
    "special_features": "scars, accessories, weapons, unique traits",
}

CHARACTER_EXTRACTION_PROMPT = """
You are a visual character designer analyzing a story text.

Read the following story segments and extract ALL named characters mentioned.
For each character, provide visual details ONLY (no personality, no backstory).

Story segments:
{text_sample}

Output a JSON object where keys are character names and values match this schema:
{schema}

Focus on appearance that would help an artist draw the character consistently.
Include only characters with at least 2 visual details mentioned.
""".strip()

PROMPT_TEMPLATE = (
    "{gender}, approximately {age} years old, {hair} hair, {eyes} eyes, "
    "{body_type}, wearing {clothing}. {special_features}"
)


class CharacterAgent(BaseAgent):
    """
    Extracts characters from segmented novel text and writes
    JSON + prompt.txt files for downstream image generation.
    """

    def __init__(self, project_name: str, episode: int = 1, **kwargs):
        super().__init__(project_name, **kwargs)
        self.episode = episode

    def _load_segments(self) -> List[Dict]:
        seg_path = self.get_path("drafts", f"episode_{self.episode}", "step1_segments.json")
        if not seg_path.exists():
            raise FileNotFoundError(
                f"Run StoryAgent first. Missing: {seg_path}"
            )
        with open(seg_path, encoding="utf-8") as f:
            return json.load(f)

    def _extract_characters(self, segments: List[Dict]) -> Dict[str, Dict]:
        """Use Qwen2.5 to extract character descriptions from segment text."""
        # Sample up to 80 segments for extraction context
        sample = segments[:80]
        text_sample = "\n".join(
            f"[{s['segment_id']}] {s['novel_text']}" for s in sample
        )

        prompt = CHARACTER_EXTRACTION_PROMPT.format(
            text_sample=text_sample,
            schema=json.dumps(CHARACTER_SCHEMA, ensure_ascii=False, indent=2),
        )

        self.log("Extracting characters with Qwen2.5 …")
        result = self.qwen.generate_json(prompt, temperature=0.1, thinking=True)

        # Normalize: result may be {name: {fields}} or [{name: ..., ...}]
        chars: Dict[str, Dict] = {}
        if isinstance(result, dict):
            for name, data in result.items():
                if isinstance(data, dict):
                    chars[name] = data
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict) and "name" in item:
                    chars[item.pop("name")] = item

        return chars

    def _build_prompt_text(self, char: Dict) -> str:
        """Build canonical prompt.txt content from character JSON."""
        parts = []
        if char.get("gender") and char["gender"] != "unknown":
            parts.append(char["gender"])
        if char.get("age"):
            parts.append(f"approximately {char['age']} years old")
        if char.get("hair"):
            parts.append(f"{char['hair']} hair")
        if char.get("eyes"):
            parts.append(f"{char['eyes']} eyes")
        if char.get("body_type"):
            parts.append(char["body_type"])
        if char.get("clothing"):
            parts.append(f"wearing {char['clothing']}")
        if char.get("special_features"):
            parts.append(char["special_features"])
        return ", ".join(parts)

    def _safe_filename(self, name: str) -> str:
        """Convert character name to a safe filename stem."""
        return re.sub(r"[^\w\-]", "_", name.strip()).lower()

    def _write_character_files(self, name: str, char_data: Dict) -> Dict[str, Path]:
        """Legacy compatibility: Write .json and _prompt.txt for a character."""
        chars_dir = self.ensure_dir("characters")
        stem = self._safe_filename(name)

        # hero.json
        json_path = chars_dir / f"{stem}.json"
        payload = {"name": name, **{k: char_data.get(k, "") for k in CHARACTER_SCHEMA}}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # hero_prompt.txt — canonical prompt fragment
        prompt_text = self._build_prompt_text(char_data)
        prompt_path = chars_dir / f"{stem}_prompt.txt"
        prompt_path.write_text(prompt_text, encoding="utf-8")

        return {"json": json_path, "prompt": prompt_path, "stem": stem}

    def run(self, overwrite: bool = False) -> Dict[str, Any]:
        """
        Extract characters and write character files.

        Note: Reference portraits (hero_ref.png) are generated during
        Stage 2 (image generation) to avoid loading both Qwen + FLUX
        simultaneously.

        Returns
        -------
        dict: character_count, characters (name → paths), project updated
        """
        segments = self._load_segments()
        existing = list(self.get_path("characters").glob("*.json")) if not overwrite else []
        existing_names = {p.stem for p in existing}

        char_data = self._extract_characters(segments)
        self.log(f"Extracted {len(char_data)} characters")

        written = {}
        new_for_pm: Dict[str, Dict] = {}

        world = WorldEngine(self.project_name, str(self.project_dir.parent))
        world.load()

        for name, data in char_data.items():
            stem = self._safe_filename(name)
            if stem in existing_names and not overwrite:
                self.log(f"  Skipping existing: {name}")
                continue

            # Update world state
            world.register_character(name, data, overwrite=overwrite)
            
            # Legacy fallback
            paths = self._write_character_files(name, data)
            written[name] = paths
            self.log(f"  Registered: {name} in WorldEngine (and {paths['json'].name})")

            new_for_pm[name] = {
                "description": self._build_prompt_text(data),
                "voice_style": "",
            }

        # Sync new characters to project.json
        if new_for_pm:
            try:
                self.pm.add_characters_batch(self.project_name, new_for_pm)
                self.log(f"Synced {len(new_for_pm)} new characters to project.json")
            except AttributeError:
                self.log("Note: ProjectManager.add_characters_batch not available — skipping sync")

        # Write step2 intermediate draft
        draft_path = self.get_path("drafts", f"episode_{self.episode}", "step2_characters.json")
        draft_path.parent.mkdir(parents=True, exist_ok=True)
        with open(draft_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    name: {
                        "stem": self._safe_filename(name),
                        "json": str(data),
                        "prompt": self._build_prompt_text(data),
                    }
                    for name, data in char_data.items()
                },
                f, ensure_ascii=False, indent=2,
            )

        world.save_if_dirty()

        return {
            "character_count": len(char_data),
            "new_count": len(written),
            "characters": {n: str(p["json"]) for n, p in written.items()},
            "draft_path": str(draft_path),
        }

    def load_character_prompt(self, character_name: str) -> str:
        """
        Load the canonical prompt text for a character.
        Used by image_generator to inject into scene prompts.
        Returns empty string if not found.
        """
        stem = self._safe_filename(character_name)
        prompt_path = self.get_path("characters", f"{stem}_prompt.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return ""

    def load_character_json(self, character_name: str) -> Dict:
        """Load character JSON dict. Returns empty dict if not found."""
        stem = self._safe_filename(character_name)
        json_path = self.get_path("characters", f"{stem}.json")
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_ref_image_path(self, character_name: str) -> Optional[Path]:
        """Return path to character reference portrait, or None if not generated yet."""
        stem = self._safe_filename(character_name)
        ref_path = self.get_path("characters", f"{stem}_ref.png")
        return ref_path if ref_path.exists() else None

    def build_character_context(self, character_names: List[str]) -> str:
        """
        Build the character context string for injection into image prompts.
        Uses WorldEngine for accurate current state.
        """
        world = WorldEngine(self.project_name, str(self.project_dir.parent)).load()
        return world.get_character_context(character_names)

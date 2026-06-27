"""
lib/agents/clue_agent.py — Scene location and prop extraction.

Ports: .claude/agents/novel-to-narration-script.md Step 2 (clues section)

Extracts named locations and key props from segment text using Qwen2.5,
writes them to projects/{name}/clues/, and syncs to project.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from lib.agents import BaseAgent

CLUE_SCHEMA = {
    "name": "location or prop name",
    "type": "location OR prop",
    "importance": "major OR minor",
    "visual_description": "concrete visual description for image generation",
    "spatial_structure": "layout / size / key visual elements (for locations)",
    "lighting": "light quality, color temperature, mood",
    "color_palette": "dominant colors",
}

CLUE_EXTRACTION_PROMPT = """
You are a visual production designer analyzing a story.

Read these story segments and extract all notable LOCATIONS and PROPS.

Focus ONLY on visual properties useful for image generation.
Do NOT include personality, symbolism, or plot significance.

Story segments:
{text_sample}

Output a JSON object where keys are the location/prop names and values match:
{schema}

Only include items that appear in at least 2 segments OR are clearly important.
""".strip()


class ClueAgent(BaseAgent):
    """Extracts scene locations and key props from segmented novel text."""

    def __init__(self, project_name: str, episode: int = 1, **kwargs):
        super().__init__(project_name, **kwargs)
        self.episode = episode

    def _load_segments(self) -> List[Dict]:
        seg_path = self.get_path("drafts", f"episode_{self.episode}", "step1_segments.json")
        if not seg_path.exists():
            raise FileNotFoundError(f"Run StoryAgent first. Missing: {seg_path}")
        with open(seg_path, encoding="utf-8") as f:
            return json.load(f)

    def _safe_filename(self, name: str) -> str:
        return re.sub(r"[^\w\-]", "_", name.strip()).lower()

    def _extract_clues(self, segments: List[Dict]) -> Dict[str, Dict]:
        sample = segments[:80]
        text_sample = "\n".join(f"[{s['segment_id']}] {s['novel_text']}" for s in sample)

        prompt = CLUE_EXTRACTION_PROMPT.format(
            text_sample=text_sample,
            schema=json.dumps(CLUE_SCHEMA, ensure_ascii=False, indent=2),
        )

        self.log("Extracting locations/props with Qwen2.5 …")
        result = self.qwen.generate_json(prompt, temperature=0.1, thinking=True)

        clues: Dict[str, Dict] = {}
        if isinstance(result, dict):
            for name, data in result.items():
                if isinstance(data, dict):
                    clues[name] = data
        return clues

    def run(self, overwrite: bool = False) -> Dict[str, Any]:
        segments = self._load_segments()
        clues_dir = self.ensure_dir("clues")

        clue_data = self._extract_clues(segments)
        self.log(f"Extracted {len(clue_data)} locations/props")

        written = {}
        new_for_pm: Dict[str, Dict] = {}

        for name, data in clue_data.items():
            stem = self._safe_filename(name)
            json_path = clues_dir / f"{stem}.json"

            if json_path.exists() and not overwrite:
                self.log(f"  Skipping existing: {name}")
                continue

            payload = {"name": name, **{k: data.get(k, "") for k in CLUE_SCHEMA}}
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            written[name] = str(json_path)
            self.log(f"  Wrote: {name} → {json_path.name}")

            new_for_pm[name] = {
                "type": data.get("type", "location"),
                "description": data.get("visual_description", ""),
                "importance": data.get("importance", "minor"),
            }

        if new_for_pm:
            try:
                self.pm.add_clues_batch(self.project_name, new_for_pm)
                self.log(f"Synced {len(new_for_pm)} new clues to project.json")
            except AttributeError:
                self.log("Note: ProjectManager.add_clues_batch not available — skipping sync")

        draft_path = self.get_path("drafts", f"episode_{self.episode}", "step2_clues.json")
        with open(draft_path, "w", encoding="utf-8") as f:
            json.dump(clue_data, f, ensure_ascii=False, indent=2)

        return {
            "clue_count": len(clue_data),
            "new_count": len(written),
            "clues": written,
            "draft_path": str(draft_path),
        }

    def get_location_description(self, location_name: str) -> str:
        """Return visual description for a location, or empty string."""
        stem = self._safe_filename(location_name)
        json_path = self.get_path("clues", f"{stem}.json")
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            parts = [data.get("visual_description", ""), data.get("lighting", "")]
            return ". ".join(p for p in parts if p)
        return ""

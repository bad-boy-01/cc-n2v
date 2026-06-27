"""
lib/script_generator.py — Script generator (v2: Gemini → QwenClient)

Reads Step 1/2 Markdown intermediate files and generates final JSON script
using the local QwenClient (Qwen2.5-7B) instead of the former Gemini API.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from pydantic import ValidationError

from lib.config import DEFAULT_LLM
from lib.qwen_client import get_qwen_client
from lib.prompt_builders_script import (
    build_drama_prompt,
    build_narration_prompt,
)
from lib.script_models import (
    DramaEpisodeScript,
    NarrationEpisodeScript,
)


class ScriptGenerator:
    """
    Script generator — reads segmented novel text and produces a structured
    JSON episode script using QwenClient (local, free, Kaggle-compatible).

    Replaces the former GeminiClient-based implementation.
    All Pydantic validation and prompt-builder logic is preserved.
    """

    def __init__(self, project_path: Union[str, Path], llm: str = DEFAULT_LLM):
        """
        Initialize generator.

        Parameters
        ----------
        project_path : str | Path
            Project directory path, e.g. projects/test0205
        llm : str
            LLM backend key: "qwen2.5-7b" (default) or "deepseek"
        """
        self.project_path = Path(project_path)
        self.client = get_qwen_client(llm=llm, load_in_4bit=True)

        # Load project.json
        self.project_json = self._load_project_json()
        self.content_mode = self.project_json.get("content_mode", "narration")

    def generate(
        self,
        episode: int,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate episode script.

        Parameters
        ----------
        episode : int
            Episode number
        output_path : Path, optional
            Output path; defaults to scripts/episode_{episode}.json

        Returns
        -------
        Path
            Path to the generated JSON file
        """
        # 1. Load intermediate file
        step1_md = self._load_step1(episode)

        # 2. Extract characters and clues from project.json
        characters = self.project_json.get("characters", {})
        clues = self.project_json.get("clues", {})

        # 3. Build prompt
        if self.content_mode == "narration":
            prompt = build_narration_prompt(
                project_overview=self.project_json.get("overview", {}),
                style=self.project_json.get("style", ""),
                style_description=self.project_json.get("style_description", ""),
                characters=characters,
                clues=clues,
                segments_md=step1_md,
            )
            schema = NarrationEpisodeScript.model_json_schema()
        else:
            prompt = build_drama_prompt(
                project_overview=self.project_json.get("overview", {}),
                style=self.project_json.get("style", ""),
                style_description=self.project_json.get("style_description", ""),
                characters=characters,
                clues=clues,
                scenes_md=step1_md,
            )
            schema = DramaEpisodeScript.model_json_schema()

        # 4. Call QwenClient (local inference — no API key, no paid service)
        print(f"📝 Generating episode {episode} script with QwenClient …")
        schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
        full_prompt = (
            prompt
            + f"\n\nOutput ONLY valid JSON matching this schema:\n{schema_str}"
        )
        result = self.client.generate_json(
            full_prompt,
            temperature=0.1,
            thinking=True,
        )

        # 5. Parse and validate
        script_data = self._validate_response(result, episode)

        # 6. Add metadata
        script_data = self._add_metadata(script_data, episode)

        # 7. Save
        if output_path is None:
            output_path = self.project_path / "scripts" / f"episode_{episode}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Script saved to {output_path}")
        return output_path

    def build_prompt(self, episode: int) -> str:
        """Build prompt only (dry-run mode)."""
        step1_md = self._load_step1(episode)
        characters = self.project_json.get("characters", {})
        clues = self.project_json.get("clues", {})

        if self.content_mode == "narration":
            return build_narration_prompt(
                project_overview=self.project_json.get("overview", {}),
                style=self.project_json.get("style", ""),
                style_description=self.project_json.get("style_description", ""),
                characters=characters,
                clues=clues,
                segments_md=step1_md,
            )
        return build_drama_prompt(
            project_overview=self.project_json.get("overview", {}),
            style=self.project_json.get("style", ""),
            style_description=self.project_json.get("style_description", ""),
            characters=characters,
            clues=clues,
            scenes_md=step1_md,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_project_json(self) -> dict:
        path = self.project_path / "project.json"
        if not path.exists():
            raise FileNotFoundError(f"project.json not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_step1(self, episode: int) -> str:
        drafts_path = self.project_path / "drafts" / f"episode_{episode}"
        if self.content_mode == "narration":
            primary = drafts_path / "step1_segments.md"
            fallback = drafts_path / "step1_normalized_script.md"
        else:
            primary = drafts_path / "step1_normalized_script.md"
            fallback = drafts_path / "step1_segments.md"

        if not primary.exists():
            if fallback.exists():
                print(f"⚠️ Using fallback: {fallback}")
                primary = fallback
            else:
                raise FileNotFoundError(f"Step 1 file not found: {primary}")

        with open(primary, "r", encoding="utf-8") as f:
            return f.read()

    def _validate_response(self, data: dict, episode: int) -> dict:
        """Validate LLM response against Pydantic model."""
        try:
            if self.content_mode == "narration":
                validated = NarrationEpisodeScript.model_validate(data)
            else:
                validated = DramaEpisodeScript.model_validate(data)
            return validated.model_dump()
        except ValidationError as e:
            print(f"⚠️ Validation warning: {e}")
            return data if isinstance(data, dict) else {}

    def _add_metadata(self, script_data: dict, episode: int) -> dict:
        """Attach episode metadata."""
        script_data.setdefault("episode", episode)
        script_data.setdefault("content_mode", self.content_mode)

        if "novel" not in script_data:
            script_data["novel"] = {
                "title": self.project_json.get("title", ""),
                "chapter": f"Episode {episode}",
                "source_file": "",
            }

        now = datetime.now().isoformat()
        script_data.setdefault("metadata", {})
        script_data["metadata"]["created_at"] = now
        script_data["metadata"]["updated_at"] = now
        script_data["metadata"]["generator"] = "QwenClient (local)"

        if self.content_mode == "narration":
            segments = script_data.get("segments", [])
            script_data["metadata"]["total_segments"] = len(segments)
            script_data["duration_seconds"] = sum(
                s.get("duration_seconds", 4) for s in segments
            )
        else:
            scenes = script_data.get("scenes", [])
            script_data["metadata"]["total_scenes"] = len(scenes)
            script_data["duration_seconds"] = sum(
                s.get("duration_seconds", 8) for s in scenes
            )

        return script_data

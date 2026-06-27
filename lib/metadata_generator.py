"""
lib/metadata_generator.py — Platform metadata generator.

Generates output/metadata.json suitable for YouTube / publishing platforms.

Content:
  - Title, description, tags
  - Chapter markers (from PlannerAgent)
  - Character list (from WorldEngine)
  - Runtime, language
  - Model stack used
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TAGS = [
    "ai-generated", "novel", "story", "ai video", "cc-novel2video",
]


class MetadataGenerator:
    """
    Platform metadata generator.

    Usage
    -----
    gen = MetadataGenerator("my_project", "projects")
    path = gen.generate(episode=1, world_engine=world, episode_plan=plan)
    """

    def __init__(
        self,
        project_name: str,
        projects_root: Optional[str] = None,
        language: str = "en",
    ):
        root = Path(projects_root) if projects_root else Path("projects")
        self.project_dir = root / project_name
        self.project_name = project_name
        self.language = language

    def generate(
        self,
        episode: int = 1,
        world_engine: Optional[Any] = None,
        episode_plan: Optional[Dict] = None,
        director_profile: Optional[Dict] = None,
        runtime_seconds: float = 0.0,
        model_stack: Optional[Dict[str, str]] = None,
    ) -> Path:
        """
        Generate and save output/metadata.json.

        Parameters
        ----------
        episode : int
        world_engine : WorldEngine, optional
        episode_plan : dict, optional
            From PlannerAgent (used for chapter markers and title)
        director_profile : dict, optional
        runtime_seconds : float
        model_stack : dict, optional
            e.g. {"llm": "qwen2.5-7b", "image": "flux_schnell", "tts": "kokoro"}

        Returns
        -------
        Path
            Path to saved metadata.json
        """
        # ── Build title ────────────────────────────────────────────────────────
        ep_plan = self._get_episode_plan(episode, episode_plan)
        ep_title = ep_plan.get("title", f"Episode {episode}") if ep_plan else f"Episode {episode}"
        project_title = self._get_project_title()
        full_title = f"{project_title} — {ep_title}"

        # ── Build description ──────────────────────────────────────────────────
        description = self._build_description(
            project_title, ep_title, episode, world_engine, director_profile
        )

        # ── Build tags ─────────────────────────────────────────────────────────
        tags = list(DEFAULT_TAGS)
        if director_profile:
            style_id = director_profile.get("id", "")
            if style_id:
                tags.append(style_id)
            if director_profile.get("tone_analysis", {}).get("dominant_themes"):
                tags.extend(director_profile["tone_analysis"]["dominant_themes"][:3])
        if world_engine:
            # Add character names as tags
            chars = list(world_engine._world.get("characters", {}).keys())
            tags.extend(chars[:5])

        tags = list(dict.fromkeys(tags))  # deduplicate while preserving order

        # ── Build chapter markers ──────────────────────────────────────────────
        chapters = self._build_chapters(episode, episode_plan, runtime_seconds)

        # ── Character list ─────────────────────────────────────────────────────
        characters = []
        if world_engine:
            characters = list(world_engine._world.get("characters", {}).keys())

        # ── Assemble metadata ──────────────────────────────────────────────────
        metadata: Dict[str, Any] = {
            "title": full_title,
            "description": description,
            "tags": tags,
            "chapters": chapters,
            "characters": characters,
            "runtime_seconds": round(runtime_seconds, 1),
            "runtime_formatted": self._format_duration(runtime_seconds),
            "language": self.language,
            "episode": episode,
            "generated_by": "CC-Novel2Video v2",
            "generated_at": datetime.utcnow().isoformat(),
            "model_stack": model_stack or {
                "llm": "Qwen2.5-7B-Instruct",
                "image": "FLUX.1-schnell",
                "tts": "Kokoro-82M",
            },
        }

        if director_profile:
            metadata["style"] = director_profile.get("name", director_profile.get("id", ""))
            metadata["aspect_ratio"] = director_profile.get("aspect_ratio", "16:9")

        # ── Save ──────────────────────────────────────────────────────────────
        output_dir = self.project_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "metadata.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"[MetadataGenerator] Metadata saved: {out_path}")
        return out_path

    def _get_project_title(self) -> str:
        """Load project title from project.json."""
        project_file = self.project_dir / "project.json"
        if project_file.exists():
            try:
                with open(project_file, encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("title", self.project_name.replace("_", " ").title())
            except Exception:
                pass
        return self.project_name.replace("_", " ").title()

    def _get_episode_plan(
        self,
        episode: int,
        episode_plan: Optional[Dict],
    ) -> Optional[Dict]:
        """Get episode plan for this specific episode."""
        if not episode_plan:
            # Try to load from disk
            plan_path = (
                self.project_dir / "drafts" / f"episode_{episode}" / "episode_plan.json"
            )
            if plan_path.exists():
                try:
                    with open(plan_path, encoding="utf-8") as f:
                        data = json.load(f)
                    for ep in data.get("episode_plans", []):
                        if ep.get("episode") == episode:
                            return ep
                except Exception:
                    pass
            return None

        # Search in provided plan
        for ep in episode_plan.get("episode_plans", []):
            if ep.get("episode") == episode:
                return ep
        return None

    def _build_description(
        self,
        project_title: str,
        ep_title: str,
        episode: int,
        world_engine: Optional[Any],
        director_profile: Optional[Dict],
    ) -> str:
        """Build video description."""
        style_name = ""
        if director_profile:
            style_name = director_profile.get("name", "")

        lines = [
            f"AI-generated video adaptation of '{project_title}'",
            f"Episode {episode}: {ep_title}",
            "",
        ]

        if style_name:
            lines.append(f"Visual style: {style_name}")

        if world_engine:
            chars = list(world_engine._world.get("characters", {}).keys())
            if chars:
                lines.append(f"Featuring: {', '.join(chars[:5])}")

        lines.extend([
            "",
            "Generated with CC-Novel2Video v2",
            "Models: Qwen2.5-7B | FLUX.1-schnell | Kokoro TTS",
            "100% free and open-source. No paid APIs.",
        ])

        return "\n".join(lines)

    def _build_chapters(
        self,
        episode: int,
        episode_plan: Optional[Dict],
        total_duration: float,
    ) -> List[Dict]:
        """Build chapter timestamp markers from episode plan."""
        chapters = [{"title": "Intro", "timestamp": "0:00", "timestamp_seconds": 0}]

        ep = self._get_episode_plan(episode, episode_plan)
        if ep and total_duration > 0:
            # Divide runtime by segment count for approximate chapter timestamps
            estimated_segments = ep.get("estimated_segments", 1)
            arc = ep.get("arc", "story")
            pacing = ep.get("pacing", "steady")

            # Simple: create 3 chapters based on story arc
            thirds = total_duration / 3
            chapters.append({
                "title": "Beginning" if arc == "introduction" else "Rising Action",
                "timestamp": self._format_duration(thirds),
                "timestamp_seconds": round(thirds),
            })
            chapters.append({
                "title": "Climax" if arc in ("climax", "rising_action") else "Conclusion",
                "timestamp": self._format_duration(thirds * 2),
                "timestamp_seconds": round(thirds * 2),
            })

        return chapters

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds as H:MM:SS or M:SS."""
        seconds = int(seconds)
        if seconds >= 3600:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            return f"{h}:{m:02d}:{s:02d}"
        m = seconds // 60
        s = seconds % 60
        return f"{m}:{s:02d}"

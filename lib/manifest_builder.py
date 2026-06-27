"""
lib/manifest_builder.py — Project manifest generator.

Collects all pipeline outputs into a single project_manifest.json
that documents everything generated, with statistics and quality results.

Called after Stage 4 (video composition) completes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ManifestBuilder:
    """
    Generates project_manifest.json from pipeline outputs.

    Usage
    -----
    builder = ManifestBuilder("my_project", "projects")
    manifest_path = builder.build(
        episode=1,
        pipeline_state=state._state,
        asset_db=db,
        world_engine=world,
        quality_report=report,
        analytics=analytics_data,
    )
    """

    def __init__(self, project_name: str, projects_root: Optional[str] = None):
        self.project_name = project_name
        root = Path(projects_root) if projects_root else Path("projects")
        self.project_dir = root / project_name

    def build(
        self,
        episode: int = 1,
        pipeline_state: Optional[Dict] = None,
        asset_db: Optional[Any] = None,       # AssetDatabase
        world_engine: Optional[Any] = None,   # WorldEngine
        quality_report: Optional[Any] = None, # EpisodeReport
        analytics: Optional[Dict] = None,
    ) -> Path:
        """
        Build and save project_manifest.json.

        Returns
        -------
        Path
            Path to the saved manifest file
        """
        now = datetime.utcnow().isoformat()

        manifest: Dict[str, Any] = {
            "project": self.project_name,
            "episode": episode,
            "generated_at": now,
            "generator": "CC-Novel2Video v2",
        }

        # ── Pipeline state ────────────────────────────────────────────────────
        if pipeline_state:
            manifest["pipeline_state"] = pipeline_state

        # ── Asset database ────────────────────────────────────────────────────
        if asset_db:
            db_stats = asset_db.get_stats()
            manifest["statistics"] = db_stats
            manifest["characters"] = asset_db.get_characters()
            manifest["locations"] = asset_db.get_locations()
            manifest["generated_images"] = asset_db.get_generated_images()
            manifest["failed_images"] = asset_db.get_failed_images()
            manifest["cached_images"] = asset_db.get_cached_images()
        else:
            # Fallback: scan project directory directly
            manifest.update(self._scan_project_dir(episode))

        # ── World engine snapshot ─────────────────────────────────────────────
        if world_engine:
            manifest["world_snapshot"] = {
                "character_count": world_engine.character_count,
                "location_count": world_engine.location_count,
                "current_chapter": world_engine._world.get("current_chapter", 1),
                "timeline_events": len(world_engine._world.get("timeline", [])),
            }

        # ── Quality report ────────────────────────────────────────────────────
        if quality_report:
            manifest["quality_summary"] = {
                "images_passed": quality_report.images_passed,
                "images_total": len(quality_report.images),
                "audio_passed": quality_report.audio_passed,
                "audio_total": len(quality_report.audio),
                "failed_scenes": [
                    Path(r.path).stem for r in quality_report.failed_images
                ],
            }

        # ── Analytics ─────────────────────────────────────────────────────────
        if analytics:
            manifest["analytics"] = analytics

        # ── Output files ──────────────────────────────────────────────────────
        output_dir = self.project_dir / "output"
        output_video = output_dir / f"episode_{episode}.mp4"
        thumbnail = output_dir / "thumbnail.png"
        metadata_file = output_dir / "metadata.json"

        manifest["output"] = {
            "video": str(output_video) if output_video.exists() else None,
            "thumbnail": str(thumbnail) if thumbnail.exists() else None,
            "metadata": str(metadata_file) if metadata_file.exists() else None,
        }

        # ── Save ──────────────────────────────────────────────────────────────
        manifest_path = self.project_dir / "project_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        logger.info(f"[ManifestBuilder] Manifest saved: {manifest_path}")
        return manifest_path

    def _scan_project_dir(self, episode: int) -> Dict[str, Any]:
        """Fallback: scan project directory if AssetDatabase not available."""
        images_dir = self.project_dir / "images"
        audio_dir = self.project_dir / "audio"
        motion_dir = self.project_dir / "cache" / "motion"
        chars_dir = self.project_dir / "characters"

        image_files = sorted(str(p) for p in images_dir.glob("*.png")) if images_dir.exists() else []
        audio_files = sorted(str(p) for p in audio_dir.glob("*.wav")) if audio_dir.exists() else []
        motion_files = sorted(str(p) for p in motion_dir.glob("*.mp4")) if motion_dir.exists() else []
        char_files = sorted(str(p) for p in chars_dir.glob("*.json")) if chars_dir.exists() else []

        return {
            "statistics": {
                "images_generated": len(image_files),
                "audio_files": len(audio_files),
                "motion_clips": len(motion_files),
                "characters": len(char_files),
            },
            "image_files": image_files[:50],  # cap for large projects
            "audio_files": audio_files[:50],
        }

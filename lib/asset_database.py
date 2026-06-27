"""
lib/asset_database.py — Structured registry of all project assets.

Tracks every generated asset (images, audio, characters, locations) with
metadata, quality results, cache status, and failure reasons.

Storage: projects/{name}/asset_database.json

This replaces scattered progress tracking across individual modules and
provides a single source of truth for ManifestBuilder and the Review page.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB: Dict[str, List] = {
    "characters": [],
    "locations": [],
    "props": [],
    "backgrounds": [],
    "music": [],
    "voices": [],
    "generated_images": [],
    "failed_images": [],
    "cached_images": [],
    "audio_files": [],
    "motion_clips": [],
    "subtitle_files": [],
}


class AssetDatabase:
    """
    Structured asset registry for a single project.

    All pipeline stages call this to register their outputs.
    ManifestBuilder and ReviewGenerator read from this.

    Usage
    -----
    db = AssetDatabase(project_dir)
    db.load()

    db.register_image("scene_001", "/path/to/img.png", backend="flux", quality="pass", size_kb=892)
    db.register_failed("scene_005", reason="OOM", retry_count=1)
    db.save()
    """

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self._path = self.project_dir / "asset_database.json"
        self._db: Dict[str, List] = {}
        self._dirty = False

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> "AssetDatabase":
        """Load from disk. Returns self for chaining."""
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._db = json.load(f)
                logger.debug(f"[AssetDatabase] Loaded: {self._path}")
            except Exception as e:
                logger.warning(f"[AssetDatabase] Load failed, using empty DB: {e}")
                self._db = json.loads(json.dumps(_DEFAULT_DB))
        else:
            self._db = json.loads(json.dumps(_DEFAULT_DB))
        return self

    def save(self) -> None:
        """Persist to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._db, f, ensure_ascii=False, indent=2)
        self._dirty = False

    def save_if_dirty(self) -> None:
        if self._dirty:
            self.save()

    # ── Image registration ────────────────────────────────────────────────────

    def register_image(
        self,
        scene_id: str,
        path: str,
        backend: str = "flux",
        quality: str = "pass",
        size_kb: float = 0.0,
        difficulty: str = "medium",
        prompt_tokens: int = 0,
    ) -> None:
        """Register a successfully generated scene image."""
        self._db.setdefault("generated_images", [])
        # Remove old entry for same scene_id if exists
        self._db["generated_images"] = [
            r for r in self._db["generated_images"] if r.get("scene_id") != scene_id
        ]
        self._db["generated_images"].append({
            "scene_id": scene_id,
            "path": str(path),
            "backend": backend,
            "quality": quality,
            "size_kb": round(size_kb, 1),
            "difficulty": difficulty,
            "prompt_tokens": prompt_tokens,
            "timestamp": _now(),
        })
        # Remove from failed list if it was there
        self._db["failed_images"] = [
            r for r in self._db.get("failed_images", []) if r.get("scene_id") != scene_id
        ]
        self._dirty = True

    def register_failed(
        self,
        scene_id: str,
        reason: str,
        retry_count: int = 0,
    ) -> None:
        """Register a failed image generation attempt."""
        self._db.setdefault("failed_images", [])
        existing = next(
            (r for r in self._db["failed_images"] if r.get("scene_id") == scene_id),
            None
        )
        if existing:
            existing["retry_count"] = retry_count
            existing["reason"] = reason
            existing["timestamp"] = _now()
        else:
            self._db["failed_images"].append({
                "scene_id": scene_id,
                "reason": reason,
                "retry_count": retry_count,
                "timestamp": _now(),
            })
        self._dirty = True

    def register_cached(
        self,
        scene_id: str,
        cache_key: str,
        source_scene_id: str,
        path: str = "",
    ) -> None:
        """Register a cache hit (image reused from cache)."""
        self._db.setdefault("cached_images", [])
        self._db["cached_images"].append({
            "scene_id": scene_id,
            "cache_key": cache_key,
            "source": source_scene_id,
            "path": str(path),
            "timestamp": _now(),
        })
        self._dirty = True

    # ── Character / Location registration ────────────────────────────────────

    def register_character(
        self,
        name: str,
        json_path: str = "",
        ref_image: str = "",
        prompt: str = "",
        history_path: str = "",
    ) -> None:
        """Register or update a character entry."""
        self._db.setdefault("characters", [])
        existing = next(
            (c for c in self._db["characters"] if c.get("name") == name),
            None
        )
        entry = {
            "name": name,
            "json_path": str(json_path),
            "ref_image": str(ref_image),
            "prompt": prompt[:200],
            "history_path": str(history_path),
            "registered_at": _now(),
        }
        if existing:
            existing.update(entry)
        else:
            self._db["characters"].append(entry)
        self._dirty = True

    def register_location(
        self,
        name: str,
        description: str = "",
        image: str = "",
    ) -> None:
        """Register or update a location entry."""
        self._db.setdefault("locations", [])
        existing = next(
            (l for l in self._db["locations"] if l.get("name") == name),
            None
        )
        entry = {"name": name, "description": description[:300], "image": str(image)}
        if existing:
            existing.update(entry)
        else:
            self._db["locations"].append(entry)
        self._dirty = True

    # ── Audio / Motion / Subtitle registration ────────────────────────────────

    def register_audio(
        self,
        scene_id: str,
        path: str,
        duration_s: float,
        voice: str = "",
        quality: str = "pass",
    ) -> None:
        """Register a synthesized audio file."""
        self._db.setdefault("audio_files", [])
        self._db["audio_files"] = [
            r for r in self._db["audio_files"] if r.get("scene_id") != scene_id
        ]
        self._db["audio_files"].append({
            "scene_id": scene_id,
            "path": str(path),
            "duration_s": round(duration_s, 2),
            "voice": voice,
            "quality": quality,
            "timestamp": _now(),
        })
        self._dirty = True

    def register_motion_clip(
        self,
        scene_id: str,
        path: str,
        duration_s: float,
        camera_motion: str = "",
    ) -> None:
        """Register a rendered motion clip."""
        self._db.setdefault("motion_clips", [])
        self._db["motion_clips"] = [
            r for r in self._db["motion_clips"] if r.get("scene_id") != scene_id
        ]
        self._db["motion_clips"].append({
            "scene_id": scene_id,
            "path": str(path),
            "duration_s": round(duration_s, 2),
            "camera_motion": camera_motion,
            "timestamp": _now(),
        })
        self._dirty = True

    def register_voice(self, voice_id: str, language: str = "en") -> None:
        """Register a TTS voice used in this project."""
        self._db.setdefault("voices", [])
        if not any(v.get("voice_id") == voice_id for v in self._db["voices"]):
            self._db["voices"].append({"voice_id": voice_id, "language": language})
            self._dirty = True

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_generated_images(self) -> List[Dict]:
        return self._db.get("generated_images", [])

    def get_failed_images(self) -> List[Dict]:
        return self._db.get("failed_images", [])

    def get_cached_images(self) -> List[Dict]:
        return self._db.get("cached_images", [])

    def get_characters(self) -> List[Dict]:
        return self._db.get("characters", [])

    def get_locations(self) -> List[Dict]:
        return self._db.get("locations", [])

    def is_image_generated(self, scene_id: str) -> bool:
        return any(
            r.get("scene_id") == scene_id
            for r in self._db.get("generated_images", [])
        )

    # ── Statistics ────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics for ManifestBuilder."""
        generated = self._db.get("generated_images", [])
        failed = self._db.get("failed_images", [])
        cached = self._db.get("cached_images", [])
        audio = self._db.get("audio_files", [])
        motion = self._db.get("motion_clips", [])

        total_images = len(generated) + len(cached)
        cache_rate = round(len(cached) / total_images, 3) if total_images > 0 else 0.0

        total_audio_duration = sum(a.get("duration_s", 0) for a in audio)

        backend_counts: Dict[str, int] = {}
        for img in generated:
            b = img.get("backend", "unknown")
            backend_counts[b] = backend_counts.get(b, 0) + 1

        return {
            "total_images": total_images,
            "images_generated": len(generated),
            "images_cached": len(cached),
            "images_failed": len(failed),
            "cache_hit_rate": cache_rate,
            "audio_files": len(audio),
            "total_audio_duration_s": round(total_audio_duration, 1),
            "motion_clips": len(motion),
            "characters": len(self._db.get("characters", [])),
            "locations": len(self._db.get("locations", [])),
            "backend_usage": backend_counts,
        }

    def full_dump(self) -> Dict:
        """Return the full database (for ManifestBuilder)."""
        return dict(self._db)


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    import datetime
    return datetime.datetime.utcnow().isoformat()

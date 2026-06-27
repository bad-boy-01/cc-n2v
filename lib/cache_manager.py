"""
lib/cache_manager.py — Unified cache for all expensive pipeline operations.

Replaces the ad-hoc scene_cache/ directory in ImageGenerator with a single,
namespaced, integrity-checked cache layer that all pipeline stages share.

Cache survives Kaggle session restarts. All keys are SHA-256 hashes of
canonical inputs so cache hits are content-addressed, not path-based.

Namespaces
----------
storyboards : scene images from StoryboardAgent
characters  : character portrait PNGs
images      : episode scene images from ImageGenerator
motion      : rendered motion clips (.mp4)
audio       : synthesized audio files (.wav)
ocr         : OCR text extraction results (.txt)
panels      : detected manhwa panel crops
subtitles   : generated .srt / .ass files
embeddings  : future character embeddings (.pt)
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

NAMESPACES = [
    "storyboards", "characters", "images", "motion",
    "audio", "ocr", "panels", "subtitles", "embeddings",
]


class CacheManager:
    """
    Unified content-addressed cache for all pipeline operations.

    Usage
    -----
    cache = CacheManager("cache")
    key = cache.make_key(location="forest", characters=["Hero"], scene_type="battle")

    if cache.exists("images", key):
        path = cache.get("images", key)
    else:
        # generate image ...
        cache.put("images", key, generated_path)
    """

    def __init__(self, cache_root: str = "cache", enabled: bool = True):
        """
        Parameters
        ----------
        cache_root : str
            Root directory for all cached files (default: "cache")
        enabled : bool
            Set False to disable caching (all hits return None)
        """
        self.root = Path(cache_root)
        self.enabled = enabled
        self._hits: Dict[str, int] = {ns: 0 for ns in NAMESPACES}
        self._misses: Dict[str, int] = {ns: 0 for ns in NAMESPACES}
        self._init_dirs()

    def _init_dirs(self) -> None:
        """Create namespace directories if they don't exist."""
        for ns in NAMESPACES:
            (self.root / ns).mkdir(parents=True, exist_ok=True)

    def _ns_dir(self, namespace: str) -> Path:
        if namespace not in NAMESPACES:
            raise ValueError(
                f"Unknown cache namespace: '{namespace}'. "
                f"Valid: {NAMESPACES}"
            )
        return self.root / namespace

    # ── Key generation ────────────────────────────────────────────────────────

    @staticmethod
    def make_key(**kwargs) -> str:
        """
        Generate a SHA-256 cache key from keyword arguments.

        All values are JSON-serialized and sorted for determinism.

        Example
        -------
        key = CacheManager.make_key(
            location="forest",
            characters=["Hero", "Mentor"],
            scene_type="dialogue"
        )
        """
        # Sort lists inside kwargs for stable hashing
        normalized = {}
        for k, v in sorted(kwargs.items()):
            if isinstance(v, list):
                normalized[k] = sorted(str(x) for x in v)
            else:
                normalized[k] = str(v).lower().strip()

        raw = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @staticmethod
    def make_text_key(text: str) -> str:
        """Generate cache key from arbitrary text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    # ── Core operations ───────────────────────────────────────────────────────

    def exists(self, namespace: str, key: str) -> bool:
        """Return True if a valid cached file exists for this key."""
        if not self.enabled:
            return False
        ns_dir = self._ns_dir(namespace)
        matches = list(ns_dir.glob(f"{key}.*"))
        return bool(matches) and all(p.stat().st_size > 0 for p in matches)

    def get(self, namespace: str, key: str) -> Optional[Path]:
        """
        Return path to cached file, or None if not cached.

        Records cache hit/miss statistics.
        """
        if not self.enabled:
            self._misses[namespace] = self._misses.get(namespace, 0) + 1
            return None

        ns_dir = self._ns_dir(namespace)
        matches = list(ns_dir.glob(f"{key}.*"))
        valid = [p for p in matches if p.stat().st_size > 0]

        if valid:
            self._hits[namespace] = self._hits.get(namespace, 0) + 1
            logger.debug(f"Cache HIT  [{namespace}] {key}")
            return valid[0]

        self._misses[namespace] = self._misses.get(namespace, 0) + 1
        logger.debug(f"Cache MISS [{namespace}] {key}")
        return None

    def put(
        self,
        namespace: str,
        key: str,
        source_path: Path,
        copy: bool = True,
    ) -> Path:
        """
        Store a file in the cache.

        Parameters
        ----------
        namespace : str
        key : str
        source_path : Path
            File to cache
        copy : bool
            If True (default), copy source to cache. If False, move it.

        Returns
        -------
        Path
            Path to the cached file
        """
        if not self.enabled:
            return source_path

        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"Cannot cache non-existent file: {source_path}")

        ns_dir = self._ns_dir(namespace)
        dest = ns_dir / f"{key}{source_path.suffix}"

        if not dest.exists():
            if copy:
                shutil.copy2(source_path, dest)
            else:
                shutil.move(str(source_path), dest)
            logger.debug(f"Cache PUT  [{namespace}] {key} ← {source_path.name}")

        return dest

    def invalidate(self, namespace: str, key: str) -> bool:
        """
        Remove a cached entry.

        Returns True if something was deleted.
        """
        ns_dir = self._ns_dir(namespace)
        deleted = False
        for p in ns_dir.glob(f"{key}.*"):
            p.unlink(missing_ok=True)
            deleted = True
            logger.info(f"Cache INVALIDATED [{namespace}] {key}")
        return deleted

    def invalidate_namespace(self, namespace: str) -> int:
        """Clear all entries in a namespace. Returns count deleted."""
        ns_dir = self._ns_dir(namespace)
        count = 0
        for p in ns_dir.iterdir():
            if p.is_file():
                p.unlink()
                count += 1
        logger.info(f"Cache CLEARED [{namespace}] — {count} entries removed")
        return count

    # ── Statistics & integrity ────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return hit/miss counts and disk usage per namespace."""
        result = {}
        total_size = 0
        for ns in NAMESPACES:
            ns_dir = self.root / ns
            files = list(ns_dir.glob("*")) if ns_dir.exists() else []
            size_bytes = sum(f.stat().st_size for f in files if f.is_file())
            total_size += size_bytes
            hits = self._hits.get(ns, 0)
            misses = self._misses.get(ns, 0)
            total = hits + misses
            result[ns] = {
                "hits": hits,
                "misses": misses,
                "hit_rate": round(hits / total, 3) if total > 0 else 0.0,
                "entries": len(files),
                "size_mb": round(size_bytes / (1024 * 1024), 2),
            }
        result["_total_size_mb"] = round(total_size / (1024 * 1024), 2)
        return result

    def integrity_check(self) -> List[str]:
        """
        Scan cache for corrupt entries (zero-byte files).

        Returns list of removed file paths.
        """
        removed = []
        for ns in NAMESPACES:
            ns_dir = self.root / ns
            if not ns_dir.exists():
                continue
            for p in ns_dir.iterdir():
                if p.is_file() and p.stat().st_size == 0:
                    p.unlink()
                    removed.append(str(p))
                    logger.warning(f"Removed zero-byte cache entry: {p}")
        if removed:
            logger.info(f"Cache integrity check: removed {len(removed)} corrupt entries")
        return removed

    def total_size_mb(self) -> float:
        """Return total cache disk usage in MB."""
        total = 0
        for ns in NAMESPACES:
            ns_dir = self.root / ns
            if ns_dir.exists():
                total += sum(
                    f.stat().st_size for f in ns_dir.iterdir() if f.is_file()
                )
        return round(total / (1024 * 1024), 2)

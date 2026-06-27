"""
lib/analytics.py — Pipeline analytics collector.

Tracks timing, cache rates, GPU memory, failures, and retries
throughout the pipeline, then writes output/analytics.json at the end.

Usage
-----
analytics = PipelineAnalytics("my_project", "projects")
analytics.start_stage("storyboard")
# ... run stage ...
analytics.end_stage("storyboard", scenes=120)
analytics.record_image("scene_001", backend="flux", duration_s=3.8)
analytics.record_cache_hit("images")
analytics.save()
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage."""
    name: str
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "duration_s": round(self.duration_s, 2),
            "items_processed": self.items_processed,
            "items_failed": self.items_failed,
            "notes": self.notes,
        }


class PipelineAnalytics:
    """
    Pipeline-wide analytics collector.

    Records timing, cache hits, GPU memory peaks, and failure counts
    throughout the pipeline. Writes analytics.json at the end.
    """

    def __init__(self, project_name: str, projects_root: Optional[str] = None):
        root = Path(projects_root) if projects_root else Path("projects")
        self.project_dir = root / project_name
        self.project_name = project_name
        self._run_id = datetime.utcnow().isoformat()
        self._started_at = time.time()

        self._stages: Dict[str, StageMetrics] = {}
        self._image_times: List[float] = []
        self._audio_times: List[float] = []
        self._cache_hits: Dict[str, int] = defaultdict(int)
        self._cache_misses: Dict[str, int] = defaultdict(int)
        self._failures: Dict[str, int] = defaultdict(int)
        self._retries: Dict[str, int] = defaultdict(int)
        self._gpu_vram_samples: List[float] = []
        self._model_stack: Dict[str, str] = {}

    # ── Stage tracking ────────────────────────────────────────────────────────

    def start_stage(self, stage: str) -> None:
        """Mark the start of a pipeline stage."""
        self._stages[stage] = StageMetrics(name=stage, started_at=time.time())
        logger.debug(f"[Analytics] Stage START: {stage}")

    def end_stage(
        self,
        stage: str,
        items_processed: int = 0,
        items_failed: int = 0,
        notes: str = "",
    ) -> None:
        """Mark the end of a pipeline stage and record duration."""
        if stage not in self._stages:
            self._stages[stage] = StageMetrics(name=stage)

        m = self._stages[stage]
        m.ended_at = time.time()
        if m.started_at:
            m.duration_s = m.ended_at - m.started_at
        m.items_processed = items_processed
        m.items_failed = items_failed
        m.notes = notes
        logger.debug(f"[Analytics] Stage END: {stage} ({m.duration_s:.1f}s, {items_processed} items)")

    # ── Per-asset tracking ────────────────────────────────────────────────────

    def record_image(
        self,
        scene_id: str,
        backend: str = "flux",
        duration_s: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record an image generation attempt."""
        if duration_s > 0:
            self._image_times.append(duration_s)
        if not success:
            self._failures["image"] += 1

    def record_audio(
        self,
        scene_id: str,
        duration_s: float = 0.0,
        success: bool = True,
    ) -> None:
        """Record an audio synthesis attempt."""
        if duration_s > 0:
            self._audio_times.append(duration_s)
        if not success:
            self._failures["audio"] += 1

    def record_retry(self, asset_type: str, scene_id: str = "") -> None:
        """Record a retry attempt."""
        self._retries[asset_type] += 1

    def record_cache_hit(self, namespace: str) -> None:
        """Record a cache hit."""
        self._cache_hits[namespace] += 1

    def record_cache_miss(self, namespace: str) -> None:
        """Record a cache miss."""
        self._cache_misses[namespace] += 1

    def sample_gpu_vram(self) -> Optional[float]:
        """Sample current GPU VRAM usage and record peak."""
        try:
            import torch
            if torch.cuda.is_available():
                used_gb = torch.cuda.memory_allocated() / 1e9
                self._gpu_vram_samples.append(used_gb)
                return used_gb
        except ImportError:
            pass
        return None

    def set_model_stack(self, **models: str) -> None:
        """Record the models used (e.g. llm='qwen2.5-7b', image='flux_schnell')."""
        self._model_stack.update(models)

    # ── Compute & save ────────────────────────────────────────────────────────

    def compute(self) -> Dict[str, Any]:
        """Compute final analytics dict."""
        total_duration = time.time() - self._started_at

        # Cache statistics
        total_hits = sum(self._cache_hits.values())
        total_misses = sum(self._cache_misses.values())
        total_cache_ops = total_hits + total_misses
        cache_hit_rate = round(total_hits / total_cache_ops, 3) if total_cache_ops > 0 else 0.0

        # Image stats
        avg_image_s = (
            round(sum(self._image_times) / len(self._image_times), 2)
            if self._image_times else 0.0
        )
        avg_audio_s = (
            round(sum(self._audio_times) / len(self._audio_times), 2)
            if self._audio_times else 0.0
        )

        # GPU peak
        gpu_peak = max(self._gpu_vram_samples) if self._gpu_vram_samples else None

        return {
            "run_id": self._run_id,
            "project": self.project_name,
            "total_runtime_seconds": round(total_duration, 1),
            "stages": {
                name: m.to_dict() for name, m in self._stages.items()
            },
            "image_generation": {
                "total": len(self._image_times),
                "avg_duration_s": avg_image_s,
                "total_duration_s": round(sum(self._image_times), 1),
            },
            "audio_synthesis": {
                "total": len(self._audio_times),
                "avg_duration_s": avg_audio_s,
                "total_duration_s": round(sum(self._audio_times), 1),
            },
            "cache": {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": cache_hit_rate,
                "by_namespace": {
                    ns: {
                        "hits": self._cache_hits.get(ns, 0),
                        "misses": self._cache_misses.get(ns, 0),
                    }
                    for ns in set(list(self._cache_hits.keys()) + list(self._cache_misses.keys()))
                },
            },
            "failures": dict(self._failures),
            "retries": dict(self._retries),
            "gpu": {
                "peak_vram_gb": round(gpu_peak, 2) if gpu_peak else None,
                "samples": len(self._gpu_vram_samples),
            },
            "model_stack": self._model_stack,
        }

    def save(self) -> Path:
        """Compute analytics and save to output/analytics.json."""
        data = self.compute()

        output_dir = self.project_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "analytics.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        total = data["total_runtime_seconds"]
        cache_rate = data["cache"]["hit_rate"]
        logger.info(
            f"[Analytics] Saved: {out_path} "
            f"(runtime={total:.0f}s, cache_hit={cache_rate:.0%})"
        )
        return out_path

    def print_summary(self) -> None:
        """Print a human-readable analytics summary."""
        data = self.compute()
        print("\n📊 Pipeline Analytics")
        print(f"   Total runtime:      {data['total_runtime_seconds']:.0f}s")
        print(f"   Avg image gen:      {data['image_generation']['avg_duration_s']:.1f}s")
        print(f"   Cache hit rate:     {data['cache']['hit_rate']:.0%}")
        print(f"   Failures:           {sum(data['failures'].values())}")
        print(f"   GPU peak VRAM:      {data['gpu']['peak_vram_gb'] or 'N/A'} GB")
        print()

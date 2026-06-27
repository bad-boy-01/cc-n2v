"""
lib/batch_scheduler.py — Asynchronous CPU work scheduler.

Optimizes overall pipeline time by running CPU-bound tasks (TTS, Motion, Video)
in a background thread pool while the main thread keeps the GPU saturated
with image generation (FLUX/SDXL) or LLM inference.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class BatchScheduler:
    """
    Manages background execution of CPU-bound pipeline tasks.

    Usage
    -----
    scheduler = BatchScheduler(max_workers=2)

    for batch in batches:
        # main thread (GPU bound)
        images = generator.generate_batch(batch)

        # offload CPU work to background
        for scene, image in zip(batch, images):
            scheduler.submit(
                task_id=scene["segment_id"],
                func=motion_engine.render,
                image_path=image,
                motion=scene["camera_motion"],
                duration_s=scene["duration_seconds"]
            )

    # Wait for all background tasks to finish
    results = scheduler.wait_all()
    """

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: Dict[Any, str] = {}
        self._results: Dict[str, Any] = {}
        self._errors: Dict[str, Exception] = {}

    def submit(self, task_id: str, func: Callable, *args: Any, **kwargs: Any) -> None:
        """Submit a task to the background pool."""
        future = self._executor.submit(func, *args, **kwargs)
        self._futures[future] = task_id
        logger.debug(f"[BatchScheduler] Submitted background task: {task_id}")

    def wait_all(self) -> Dict[str, Any]:
        """
        Block until all submitted tasks complete.

        Returns
        -------
        Dict[str, Any]
            Mapping of task_id to return value
        """
        logger.info(f"[BatchScheduler] Waiting for {len(self._futures)} background tasks...")

        for future in as_completed(self._futures):
            task_id = self._futures[future]
            try:
                result = future.result()
                self._results[task_id] = result
                logger.debug(f"[BatchScheduler] Task completed: {task_id}")
            except Exception as e:
                self._errors[task_id] = e
                logger.error(f"[BatchScheduler] Task failed: {task_id} - {e}")

        self._futures.clear()

        if self._errors:
            logger.warning(
                f"[BatchScheduler] {len(self._errors)} tasks failed. "
                "Check logs for details."
            )

        return self._results

    def get_errors(self) -> Dict[str, Exception]:
        """Return dict of failed task IDs and their exceptions."""
        return self._errors

    def shutdown(self) -> None:
        """Shut down the thread pool."""
        self._executor.shutdown(wait=True)

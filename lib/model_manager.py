"""
lib/model_manager.py — Centralized model lifecycle manager.

Consolidates all model loading, unloading, VRAM management, and gc calls
that were previously scattered across VideoAgent, pipeline_service.py,
and run_pipeline.py.

Usage
-----
from lib.model_manager import ModelManager

mm = ModelManager()
mm.unload_llm("qwen2.5-7b")
mm.unload_image(gen)
mm.unload_tts(tts)
mm.free_vram("between Stage 1 and Stage 2")
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Centralized model lifecycle manager.

    Owns:
      - Unloading models after each pipeline stage
      - gc.collect() + torch.cuda.empty_cache() calls
      - VRAM usage logging

    Ensures only one model family is active at a time, which is the
    critical invariant for Kaggle T4/P100 with limited VRAM.
    """

    def __init__(self) -> None:
        pass

    # ── Stage 1 teardown ──────────────────────────────────────────────────────

    def unload_llm(self, llm_id: str) -> None:
        """
        Unload Qwen/DeepSeek after Stage 1 (Storyboard).
        Safe to call even if the client was never loaded.
        """
        logger.info("[ModelManager] Unloading LLM: %s …", llm_id)
        try:
            from lib.qwen_client import get_qwen_client
            client = get_qwen_client(llm_id)
            client.unload_model()
            logger.info("[ModelManager] LLM unloaded ✅")
        except Exception as e:
            logger.warning("[ModelManager] LLM unload warning (continuing): %s", e)
        finally:
            self.free_vram(label=f"after LLM ({llm_id}) unload")

    # ── Stage 2 teardown ──────────────────────────────────────────────────────

    def unload_image(self, generator: Any) -> None:
        """
        Unload FLUX/SDXL after Stage 2 (Image Generation).

        Parameters
        ----------
        generator : ImageGenerator
            The active ImageGenerator instance to unload.
        """
        logger.info("[ModelManager] Unloading image generator …")
        try:
            generator.unload_model()
            logger.info("[ModelManager] Image generator unloaded ✅")
        except Exception as e:
            logger.warning("[ModelManager] Image generator unload warning: %s", e)
        finally:
            self.free_vram(label="after image generator unload")

    # ── Stage 3 teardown ──────────────────────────────────────────────────────

    def unload_tts(self, tts: Any) -> None:
        """
        Unload Kokoro TTS after Stage 3 (Audio).
        Kokoro runs on CPU — no VRAM to free, but gc still helps.

        Parameters
        ----------
        tts : KokoroTTS
            The active TTS instance to unload.
        """
        logger.info("[ModelManager] Unloading TTS …")
        try:
            tts.unload_model()
            logger.info("[ModelManager] TTS unloaded ✅")
        except Exception as e:
            logger.warning("[ModelManager] TTS unload warning: %s", e)
        finally:
            # Kokoro is CPU-only; collect anyway for Python object cleanup
            gc.collect()

    # ── Shared utilities ──────────────────────────────────────────────────────

    def free_vram(self, label: str = "") -> None:
        """
        Run gc.collect() + torch.cuda.empty_cache() and log current VRAM.

        Parameters
        ----------
        label : str
            Context string included in the log message (e.g., "between Stage 1 and 2").
        """
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.log_vram(label=label)
        except ImportError:
            pass  # torch not installed (CPU-only or dry-run mode)

    def log_vram(self, label: str = "") -> None:
        """
        Log current GPU memory usage.

        Parameters
        ----------
        label : str
            Context string included in the log line.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            tag = f" [{label}]" if label else ""
            logger.info(
                "[ModelManager]%s VRAM — allocated: %.2f GB, reserved: %.2f GB",
                tag, allocated_gb, reserved_gb,
            )
        except ImportError:
            pass

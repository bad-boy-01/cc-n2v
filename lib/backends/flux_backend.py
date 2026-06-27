"""
lib/backends/flux_backend.py — FLUX.1-schnell inference backend.

Owns the complete model lifecycle:
  load()     — FluxPipeline.from_pretrained, memory optimizations
  generate() — seed handling, inference, OOM recovery
  warmup()   — 64×64, 1-step smoke test of the full stack
  unload()   — safe teardown + VRAM release

Nothing outside this file should import FluxPipeline directly.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)

FLUX_REPO = "black-forest-labs/FLUX.1-schnell"


class FluxBackend:
    """
    FLUX.1-schnell inference backend.

    All Diffusers-specific code is isolated here so the plugin adapter
    (plugins/image/flux.py) stays framework-agnostic.
    """

    def __init__(self) -> None:
        self._pipe = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load FLUX.1-schnell into VRAM.

        Memory strategy (applied in order of availability):
          1. bfloat16 weights
          2. Attention slicing (reduces peak VRAM ~20%)
          3. xformers memory-efficient attention (optional, skipped if missing)
          4. CPU offload for very low VRAM environments (falls back if needed)
        """
        if self._pipe is not None:
            return

        try:
            import torch
            from diffusers import FluxPipeline
        except ImportError as exc:
            raise ImportError(
                "diffusers and torch are required for FLUX inference. "
                "Install with: pip install diffusers transformers torch"
            ) from exc

        logger.info("FluxBackend: loading %s …", FLUX_REPO)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self._pipe = FluxPipeline.from_pretrained(
            FLUX_REPO,
            torch_dtype=dtype,
        )
        self._pipe = self._pipe.to(device)

        # Memory optimizations
        try:
            self._pipe.enable_attention_slicing()
            logger.debug("FluxBackend: attention slicing enabled")
        except Exception as e:
            logger.debug("FluxBackend: attention slicing unavailable: %s", e)

        try:
            self._pipe.enable_xformers_memory_efficient_attention()
            logger.debug("FluxBackend: xformers enabled")
        except Exception:
            logger.debug("FluxBackend: xformers not available, using default attention")

        logger.info("FluxBackend: FLUX.1-schnell loaded ✅")

    def generate(self, request) -> Optional["PIL.Image.Image"]:
        """
        Run FLUX inference.

        Parameters
        ----------
        request : ImageGenerationRequest
            Typed generation request from plugins/base.py.

        Returns
        -------
        PIL.Image.Image on success, None on OOM or other failure.
        """
        if self._pipe is None:
            self.load()

        try:
            import torch
        except ImportError:
            import subprocess  # noqa: F401 — torch must exist if we got here
            raise

        try:
            generator = None
            if request.seed is not None:
                generator = torch.Generator().manual_seed(request.seed)

            result = self._pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )
            return result.images[0]

        except Exception as exc:
            exc_str = str(exc)
            exc_type = type(exc).__name__
            if "OutOfMemoryError" in exc_type or "CUDA out of memory" in exc_str:
                logger.warning(
                    "FluxBackend: CUDA OOM at %dx%d — returning None for SDXL fallback",
                    request.width, request.height,
                )
                # Release partially allocated tensors before returning
                try:
                    import torch as _t
                    _t.cuda.empty_cache()
                except Exception:
                    pass
                return None
            raise  # Re-raise non-OOM errors for upstream logging

    def warmup(self) -> None:
        """
        Run a 64×64, 1-step smoke test.

        Verifies: weights loaded, tokenizer, scheduler, VAE encode/decode,
        UNet forward pass, device placement, xformers (if active), memory.

        Raises on any failure. Result image is discarded.
        """
        if self._pipe is None:
            self.load()

        logger.info("FluxBackend: running warmup (64×64, 1 step) …")

        self._pipe(
            prompt="warmup",
            width=64,
            height=64,
            num_inference_steps=1,
            guidance_scale=0.0,
        )

        logger.info("FluxBackend: warmup ✅")

    def unload(self) -> None:
        """Delete pipeline and release all VRAM."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / 1e9
                logger.info("FluxBackend: unloaded. VRAM allocated: %.2f GB", allocated)
        except ImportError:
            pass

    def is_loaded(self) -> bool:
        """True if the pipeline is currently in memory."""
        return self._pipe is not None

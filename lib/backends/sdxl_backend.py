"""
lib/backends/sdxl_backend.py — Stable Diffusion XL inference backend.

Owns the complete model lifecycle:
  load()     — StableDiffusionXLPipeline.from_pretrained, memory optimizations
  generate() — seed handling, portrait dimension adjustment, inference, OOM recovery
  warmup()   — 64×64, 1-step smoke test of the full stack
  unload()   — safe teardown + VRAM release

SDXL is landscape-native (optimal at 1024×1024).
Portrait generation (height > width) is handled by capping to 768×1344.

Nothing outside this file should import StableDiffusionXLPipeline directly.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"

# SDXL portrait sweet spot — avoids artifacts at extreme aspect ratios
SDXL_MAX_PORTRAIT_W = 768
SDXL_MAX_PORTRAIT_H = 1344


class SDXLBackend:
    """
    Stable Diffusion XL inference backend.

    All Diffusers-specific code is isolated here so the plugin adapter
    (plugins/image/sdxl.py) stays framework-agnostic.
    """

    def __init__(self) -> None:
        self._pipe = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load Stable Diffusion XL base into VRAM.

        Memory strategy:
          1. float16 weights (SDXL ships in float16)
          2. Disable safety checker (not needed for anime/novel content)
          3. Attention slicing (reduces peak VRAM ~20%)
          4. xformers memory-efficient attention (optional)
          5. VAE slicing for large batches (optional)
        """
        if self._pipe is not None:
            return

        try:
            import torch
            from diffusers import StableDiffusionXLPipeline, AutoencoderKL
        except ImportError as exc:
            raise ImportError(
                "diffusers and torch are required for SDXL inference. "
                "Install with: pip install diffusers transformers torch"
            ) from exc

        logger.info("SDXLBackend: loading %s …", SDXL_REPO)

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        vae = None
        if dtype == torch.float16:
            try:
                logger.info("SDXLBackend: loading fp16 fixed VAE …")
                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", 
                    torch_dtype=torch.float16
                )
            except Exception as e:
                logger.warning("SDXLBackend: could not load fp16 VAE fix, will try without it: %s", e)

        self._pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_REPO,
            vae=vae,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if torch.cuda.is_available() else None,
        )
        
        self._pipe = self._pipe.to(device)

        # Disable safety checker for creative content generation
        self._pipe.safety_checker = None

        # Memory optimizations
        try:
            self._pipe.enable_attention_slicing()
            logger.debug("SDXLBackend: attention slicing enabled")
        except Exception as e:
            logger.debug("SDXLBackend: attention slicing unavailable: %s", e)

        try:
            self._pipe.enable_xformers_memory_efficient_attention()
            logger.debug("SDXLBackend: xformers enabled")
        except Exception:
            logger.debug("SDXLBackend: xformers not available, using default attention")

        try:
            self._pipe.vae.enable_slicing()
            logger.debug("SDXLBackend: VAE slicing enabled")
        except Exception:
            pass

        logger.info("SDXLBackend: SDXL loaded ✅")

    def generate(self, request) -> Optional["PIL.Image.Image"]:
        """
        Run SDXL inference.

        SDXL is landscape-native. Portrait dimensions (height > width) are
        automatically capped to 768×1344 to avoid quality degradation.

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
            raise

        # SDXL portrait dimension adjustment
        w, h = request.width, request.height
        if h > w:
            w = min(w, SDXL_MAX_PORTRAIT_W)
            h = min(h, SDXL_MAX_PORTRAIT_H)
            logger.debug("SDXLBackend: portrait mode → capped to %dx%d", w, h)

        # Use request guidance_scale if non-zero, otherwise default to 7.5 for SDXL
        guidance_scale = request.guidance_scale if request.guidance_scale > 0 else 7.5
        num_steps = request.num_steps if request.num_steps > 1 else 30

        OOM_FALLBACK_SIZES = [(512, 896), (448, 768)]
        
        for attempt_w, attempt_h in [(w, h)] + OOM_FALLBACK_SIZES:
            try:
                generator = None
                if request.seed is not None:
                    generator = torch.Generator().manual_seed(request.seed)

                result = self._pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt or None,
                    width=attempt_w,
                    height=attempt_h,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                return result.images[0]

            except Exception as exc:
                exc_str = str(exc)
                exc_type = type(exc).__name__
                if "OutOfMemoryError" in exc_type or "CUDA out of memory" in exc_str:
                    try:
                        import torch as _t
                        _t.cuda.empty_cache()
                    except Exception:
                        pass
                        
                    if (attempt_w, attempt_h) == OOM_FALLBACK_SIZES[-1]:
                        logger.warning("SDXLBackend: exhausted all OOM fallbacks, returning None")
                        return None
                    else:
                        logger.warning("SDXLBackend: CUDA OOM at %dx%d, retrying smaller...", attempt_w, attempt_h)
                        continue
                        
                elif "Expected all tensors to be on the same device" in exc_str:
                    logger.error("SDXLBackend: Device mismatch — pipeline may be contaminated by accelerate hooks. Unloading.")
                    self.unload()
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

        logger.info("SDXLBackend: running warmup (64×64, 1 step) …")

        self._pipe(
            prompt="warmup",
            width=64,
            height=64,
            num_inference_steps=1,
            guidance_scale=0.0,
        )

        logger.info("SDXLBackend: warmup ✅")

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
                logger.info("SDXLBackend: unloaded. VRAM allocated: %.2f GB", allocated)
        except ImportError:
            pass

    def is_loaded(self) -> bool:
        """True if the pipeline is currently in memory."""
        return self._pipe is not None

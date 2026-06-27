"""
plugins/image/sdxl.py — Stable Diffusion XL plugin adapter.

This file is intentionally minimal (~30 lines). All inference logic lives in
lib/backends/sdxl_backend.py. This adapter simply bridges the plugin interface
to the backend.

SDXL serves as the automatic fallback when FLUX runs out of memory.
Portrait dimension capping (768×1344) is handled inside SDXLBackend.
"""

from __future__ import annotations

from typing import Any, Dict

from plugins.base import ImageGenerationRequest, ImagePlugin
from lib.backends.sdxl_backend import SDXLBackend


class Plugin(ImagePlugin):
    """Stable Diffusion XL image generation plugin."""

    def __init__(self) -> None:
        self._backend = SDXLBackend()

    def load(self) -> None:
        self._backend.load()

    def generate(self, request: ImageGenerationRequest) -> Any:
        return self._backend.generate(request)

    def warmup(self) -> None:
        self._backend.warmup()

    def unload(self) -> None:
        self._backend.unload()

    @property
    def plugin_id(self) -> str:
        return "sdxl"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "Stable Diffusion XL",
            "family": "sdxl",
            "default_steps": 30,
            "supports_ip_adapter": True,
            "supports_lora": True,
        }

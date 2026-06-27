"""
plugins/image/flux.py — FLUX.1-schnell plugin adapter.

This file is intentionally minimal (~30 lines). All inference logic lives in
lib/backends/flux_backend.py. This adapter simply bridges the plugin interface
to the backend.

To add a new image model:
  1. Create lib/backends/<model>_backend.py
  2. Create a copy of this file pointing at the new backend
  3. Add one entry to plugins/__init__.py IMAGE_PLUGINS registry
"""

from __future__ import annotations

from typing import Any, Dict

from plugins.base import ImageGenerationRequest, ImagePlugin
from lib.backends.flux_backend import FluxBackend


class Plugin(ImagePlugin):
    """FLUX.1-schnell image generation plugin."""

    def __init__(self) -> None:
        self._backend = FluxBackend()

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
        return "flux_schnell"

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "name": "FLUX.1-schnell",
            "family": "flux",
            "default_steps": 4,
            "supports_ip_adapter": False,
            "supports_lora": True,
        }

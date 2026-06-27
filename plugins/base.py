"""
plugins/base.py — Abstract interfaces for all CC-Novel2Video plugins.

Every model-specific implementation must inherit from these interfaces.
The core pipeline only imports these base classes — never concrete plugins.

Adding a new model = add one file to plugins/{type}/.
Nothing in the core pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ImageGenerationRequest:
    """
    Typed request object for all image generation calls.

    Using a dataclass instead of positional arguments means new parameters
    (LoRA weights, IP-Adapter images, style tokens, etc.) can be added here
    without changing any plugin or backend signatures.
    """

    prompt: str
    negative_prompt: str = ""
    width: int = 1920
    height: int = 1080
    seed: Optional[int] = None
    guidance_scale: float = 0.0
    num_steps: int = 4
    character_refs: List[Path] = field(default_factory=list)
    style: str = ""
    output_path: Optional[Path] = None


class ImagePlugin(ABC):
    """
    Abstract interface for image generation backends.

    Plugins are thin adapters — they delegate all Diffusers logic to a
    backend class in lib/backends/. The plugin itself holds no model weights.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model into memory/VRAM."""

    @abstractmethod
    def generate(self, request: ImageGenerationRequest) -> Any:
        """
        Generate an image from a typed request.

        Parameters
        ----------
        request : ImageGenerationRequest
            Fully-specified generation parameters.

        Returns
        -------
        PIL.Image.Image or None on failure.
        """

    @abstractmethod
    def warmup(self) -> None:
        """
        Run a 64×64, 1-step inference to verify the entire pipeline stack.

        Tests: model weights, tokenizer, scheduler, VAE encode/decode,
        UNet forward pass, device placement, xformers (if enabled), memory.

        Expected time: 1–2 seconds. Result is discarded.
        Raises on any failure — pipeline must stop if warmup fails.
        """

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free VRAM."""

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique plugin identifier (matches plugins/__init__.py registry)."""

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Static capability descriptor. Example:
        {
            "name": "FLUX.1-schnell",
            "family": "flux",
            "default_steps": 4,
            "supports_ip_adapter": False,
            "supports_lora": True,
        }
        VRAM is intentionally excluded — actual usage varies with resolution,
        LoRA, scheduler, and batch size. Estimate dynamically if needed.
        """


class TTSPlugin(ABC):
    """Abstract interface for text-to-speech backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model into memory."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Path,
        voice: str = "",
        speed: float = 1.0,
        language: str = "en",
    ) -> Tuple[Path, float]:
        """
        Synthesize speech.

        Returns
        -------
        tuple: (output_path, duration_seconds)
        """

    @abstractmethod
    def unload(self) -> None:
        """Unload model."""

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique plugin identifier."""

    @property
    @abstractmethod
    def available_voices(self) -> List[str]:
        """List of available voice IDs."""


class LLMPlugin(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def load(self, load_in_4bit: bool = True) -> None:
        """Load model."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        thinking: bool = False,
    ) -> str:
        """Generate text response."""

    @abstractmethod
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.1,
        thinking: bool = False,
    ) -> Any:
        """Generate and parse JSON response."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free VRAM."""

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique plugin identifier."""

    @property
    @abstractmethod
    def vram_gb(self) -> float:
        """Estimated VRAM usage in GB."""


class MotionPlugin(ABC):
    """Abstract interface for motion/animation backends."""

    @abstractmethod
    def render(
        self,
        image_path: Path,
        output_path: Path,
        motion: str,
        duration_s: float,
        fps: int = 24,
        resolution: str = "1080p",
    ) -> Path:
        """Render a motion clip from a static image."""

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique plugin identifier."""


class OCRPlugin(ABC):
    """Abstract interface for OCR backends."""

    @abstractmethod
    def extract(self, image_path: Path, languages: Optional[List[str]] = None) -> str:
        """Extract text from image. Returns extracted text."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model."""

    @property
    @abstractmethod
    def plugin_id(self) -> str:
        """Unique plugin identifier."""


class InputAdapter(ABC):
    """Abstract interface for input source adapters."""

    @abstractmethod
    def read(self, source: str) -> str:
        """
        Read content from source and return as plain text.

        Parameters
        ----------
        source : str
            File path, URL, or directory path

        Returns
        -------
        str
            Plain text content
        """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of file extensions this adapter handles."""

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Unique adapter identifier."""

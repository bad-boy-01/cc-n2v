"""
plugins/__init__.py — Plugin loader for CC-Novel2Video v2.

Loads plugins by ID from models.json or by direct name.
All plugin loading is centralized here so the rest of the pipeline
never imports concrete plugin classes directly.

Usage
-----
from plugins import load_image_plugin, load_tts_plugin

img_plugin = load_image_plugin("flux_schnell")
img_plugin.load()
image = img_plugin.generate(prompt, 1920, 1080)
img_plugin.unload()
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Optional

from plugins.base import ImagePlugin, LLMPlugin, MotionPlugin, OCRPlugin, TTSPlugin

logger = logging.getLogger(__name__)

# Plugin registry: plugin_id → module path
IMAGE_PLUGINS = {
    "flux_schnell": "plugins.image.flux",
    "flux":         "plugins.image.flux",
    "sdxl":         "plugins.image.sdxl",
}

TTS_PLUGINS = {
    "kokoro":    "plugins.tts.kokoro",
    "edge-tts":  "plugins.tts.edge_tts",
    "edge_tts":  "plugins.tts.edge_tts",
}

LLM_PLUGINS = {
    "qwen2.5-7b": "plugins.llm.qwen",
    "qwen2.5":    "plugins.llm.qwen",
    "deepseek":   "plugins.llm.qwen",  # same wrapper, different model ID
}

MOTION_PLUGINS = {
    "ken_burns": "plugins.motion.ken_burns",
    "default":   "plugins.motion.ken_burns",
}

OCR_PLUGINS = {
    "easyocr":   "plugins.ocr.easyocr",
    "default":   "plugins.ocr.easyocr",
}


def load_image_plugin(plugin_id: str = "flux_schnell") -> ImagePlugin:
    """Load and return an image generation plugin."""
    return _load_plugin(IMAGE_PLUGINS, plugin_id, "image", ImagePlugin)


def load_tts_plugin(plugin_id: str = "kokoro") -> TTSPlugin:
    """Load and return a TTS plugin."""
    return _load_plugin(TTS_PLUGINS, plugin_id, "tts", TTSPlugin)


def load_llm_plugin(plugin_id: str = "qwen2.5-7b") -> LLMPlugin:
    """Load and return an LLM plugin."""
    return _load_plugin(LLM_PLUGINS, plugin_id, "llm", LLMPlugin)


def load_motion_plugin(plugin_id: str = "ken_burns") -> MotionPlugin:
    """Load and return a motion plugin."""
    return _load_plugin(MOTION_PLUGINS, plugin_id, "motion", MotionPlugin)


def load_ocr_plugin(plugin_id: str = "easyocr") -> OCRPlugin:
    """Load and return an OCR plugin."""
    return _load_plugin(OCR_PLUGINS, plugin_id, "ocr", OCRPlugin)


def list_plugins() -> dict:
    """Return all registered plugin IDs by category."""
    return {
        "image":  list(IMAGE_PLUGINS.keys()),
        "tts":    list(TTS_PLUGINS.keys()),
        "llm":    list(LLM_PLUGINS.keys()),
        "motion": list(MOTION_PLUGINS.keys()),
        "ocr":    list(OCR_PLUGINS.keys()),
    }


def _load_plugin(registry: dict, plugin_id: str, category: str, base_class: type):
    """Generic plugin loader with fallback."""
    module_path = registry.get(plugin_id)
    if not module_path:
        available = list(registry.keys())
        raise ValueError(
            f"Unknown {category} plugin: '{plugin_id}'. "
            f"Available: {available}"
        )
    try:
        module = importlib.import_module(module_path)
        plugin = module.Plugin()
        logger.info(f"[PluginLoader] Loaded {category} plugin: {plugin_id}")
        return plugin
    except ImportError as e:
        raise ImportError(
            f"Failed to import {category} plugin '{plugin_id}' from {module_path}: {e}"
        ) from e

"""
lib/config.py — Centralized configuration for CC-Novel2Video free pipeline.

All default values live here.
Override at runtime via:
  - CLI flags: --voice af_bella --resolution 1440p
  - Environment variables: CCNV_VOICE=af_bella
  - Direct import: from lib.config import DEFAULT_VOICE
"""

from __future__ import annotations

import os

# ── LLM ───────────────────────────────────────────────────────────────────────

DEFAULT_LLM = os.environ.get("CCNV_LLM", "qwen2.5-7b")
# Options: "qwen2.5-7b" | "deepseek-r1-7b"

DEFAULT_LLM_4BIT = os.environ.get("CCNV_LLM_4BIT", "true").lower() != "false"
# Set False only if you have 24+ GB VRAM

MAX_SCENES_PER_BATCH = int(os.environ.get("CCNV_BATCH_SIZE", "25"))
# Storyboard batching — prevents Qwen context overflow on long novels

# ── Image Generation ──────────────────────────────────────────────────────────

DEFAULT_IMAGE_MODEL = os.environ.get("CCNV_IMAGE_MODEL", "flux_schnell")
# Options: "flux_schnell" | "sdxl"
# flux_schnell: Apache 2.0, ~7 GB VRAM, 4 steps — recommended for T4
# sdxl: Apache 2.0, ~5 GB VRAM, 30 steps — fallback

DEFAULT_RESOLUTION = os.environ.get("CCNV_RESOLUTION", "1080p")
# Options: "1080p" | "1440p" | "4k" | "16:9"

ENABLE_IP_ADAPTER = os.environ.get("CCNV_IP_ADAPTER", "false").lower() == "true"
# IP-Adapter for character reference images
# Disabled by default — adds VRAM requirement, skip on OOM

# ── Audio / TTS ───────────────────────────────────────────────────────────────

DEFAULT_VOICE = os.environ.get("CCNV_VOICE", "af_heart")
# Kokoro voice packs:
#   af_heart  — female, warm, good for narration/audiobook (DEFAULT)
#   af_bella  — female, clear, professional narrator style
#   am_adam   — male, authoritative
#   am_michael— male, storytelling tone
# Full list: https://huggingface.co/hexgrad/Kokoro-82M

DEFAULT_TTS_SPEED = float(os.environ.get("CCNV_TTS_SPEED", "1.0"))
# 1.0 = normal, 0.9 = slightly slower (better for comprehension)

# ── Motion Engine ─────────────────────────────────────────────────────────────

MIN_SCENE_DURATION = float(os.environ.get("CCNV_MIN_SCENE_DURATION", "3.0"))
# Seconds — scene clip is never shorter than this

DEFAULT_FPS = int(os.environ.get("CCNV_FPS", "24"))
# 24 fps for cinematic feel, 30 for smoother motion

MOTION_ZOOM_FACTOR = float(os.environ.get("CCNV_ZOOM_FACTOR", "1.12"))
# Ken Burns zoom level — 1.12 = 12% zoom

# ── Subtitles ─────────────────────────────────────────────────────────────────

DEFAULT_SUBTITLE_FONT = os.environ.get("CCNV_SUB_FONT", "Arial")
DEFAULT_SUBTITLE_FONT_SIZE = int(os.environ.get("CCNV_SUB_FONT_SIZE", "40"))
DEFAULT_SUBTITLE_COLOR = os.environ.get("CCNV_SUB_COLOR", "&H00FFFFFF")   # white (ASS format)
DEFAULT_SUBTITLE_OUTLINE = os.environ.get("CCNV_SUB_OUTLINE", "&H00000000")  # black

# ── Video Composition ─────────────────────────────────────────────────────────

DEFAULT_VIDEO_CRF = int(os.environ.get("CCNV_CRF", "23"))
# H.264 CRF: 18=high quality, 23=balanced, 28=small file

DEFAULT_VIDEO_PRESET = os.environ.get("CCNV_PRESET", "fast")
# FFmpeg preset: ultrafast/fast/medium/slow

BGM_VOLUME = float(os.environ.get("CCNV_BGM_VOLUME", "0.15"))
# Background music volume relative to narration (0.0–1.0)

# ── Manhwa / OCR ──────────────────────────────────────────────────────────────

DEFAULT_READING_ORDER = os.environ.get("CCNV_READING_ORDER", "manhwa")
# "manhwa" = left-to-right (Korean webtoon)
# "manga"  = right-to-left (Japanese)

OCR_MIN_CONFIDENCE = float(os.environ.get("CCNV_OCR_CONFIDENCE", "0.4"))
# EasyOCR confidence threshold (0.0–1.0)

PANEL_DETECTION_THRESHOLD = int(os.environ.get("CCNV_PANEL_THRESHOLD", "50"))
# OpenCV contour area threshold for panel detection

# ── Kaggle Optimizations ──────────────────────────────────────────────────────

KAGGLE_MODE = os.environ.get("CCNV_KAGGLE", "false").lower() == "true"
# Enables aggressive memory management for Kaggle T4

CHECKPOINT_EVERY_N = int(os.environ.get("CCNV_CHECKPOINT_N", "1"))
# Save checkpoint every N scenes (1 = after every scene, safest)

# ── Paths ─────────────────────────────────────────────────────────────────────

DEFAULT_PROJECTS_ROOT = os.environ.get("CCNV_PROJECTS_ROOT", "projects")
DEFAULT_CACHE_ROOT = os.environ.get("CCNV_CACHE_ROOT", "cache")

# ── World Engine ──────────────────────────────────────────────────────────────

WORLD_ENGINE_ENABLED = os.environ.get("CCNV_WORLD_ENGINE", "true").lower() != "false"
# Tracks full fictional world state (characters, locations, timeline, etc.)

# ── Director / Style ──────────────────────────────────────────────────────────

DEFAULT_STYLE_PROFILE = os.environ.get("CCNV_STYLE", "cinematic")
# Options: cinematic | anime | manhwa | manga | western_comic | documentary | youtube_recap
# Loaded from styles/{profile}.json

# ── Prompt Pipeline ───────────────────────────────────────────────────────────

PROMPT_QUALITY_SUFFIX = os.environ.get(
    "CCNV_QUALITY_SUFFIX",
    "masterpiece, best quality, highly detailed, 8k, sharp focus"
)
PROMPT_MAX_TOKENS = int(os.environ.get("CCNV_PROMPT_MAX_TOKENS", "400"))
PROMPT_OPTIMIZER_ENABLED = os.environ.get("CCNV_PROMPT_OPTIMIZER", "true").lower() != "false"

# ── Scene Difficulty Routing ──────────────────────────────────────────────────

DIFFICULTY_ROUTING_ENABLED = os.environ.get("CCNV_DIFF_ROUTING", "true").lower() != "false"
# True = auto-route Easy/Medium → SDXL, Hard/Extreme → FLUX
# False = always use FLUX regardless of difficulty

# ── Cache Manager ─────────────────────────────────────────────────────────────

CACHE_ENABLED = os.environ.get("CCNV_CACHE", "true").lower() != "false"

# ── Quality Checker ───────────────────────────────────────────────────────────

QUALITY_CHECK_ENABLED = os.environ.get("CCNV_QUALITY_CHECK", "true").lower() != "false"
QUALITY_MIN_IMAGE_SIZE_KB = int(os.environ.get("CCNV_QUALITY_MIN_IMG_KB", "5"))
QUALITY_MIN_AUDIO_DURATION = float(os.environ.get("CCNV_QUALITY_MIN_AUDIO", "0.5"))

# ── Motion Planner ────────────────────────────────────────────────────────────

MOTION_PLANNER_ENABLED = os.environ.get("CCNV_MOTION_PLANNER", "true").lower() != "false"
MOTION_PLANNER_LLM_HINTS = os.environ.get("CCNV_MOTION_LLM", "false").lower() == "true"
# Set True to allow QwenClient to override rule-based camera decisions (costs tokens)

# ── Batch Scheduler ───────────────────────────────────────────────────────────

BATCH_SCHEDULER_ENABLED = os.environ.get("CCNV_BATCH_SCHED", "true").lower() != "false"
BATCH_SCHEDULER_CPU_WORKERS = int(os.environ.get("CCNV_CPU_WORKERS", "2"))
# CPU workers for parallel motion + audio while GPU generates next image batch

# ── Thumbnail Generator ───────────────────────────────────────────────────────

THUMBNAIL_ENABLED = os.environ.get("CCNV_THUMBNAIL", "true").lower() != "false"
THUMBNAIL_FONT_SIZE = int(os.environ.get("CCNV_THUMBNAIL_FONT", "72"))
THUMBNAIL_TITLE_COLOR = os.environ.get("CCNV_THUMBNAIL_COLOR", "#FFFFFF")
THUMBNAIL_WIDTH = int(os.environ.get("CCNV_THUMBNAIL_W", "1280"))
THUMBNAIL_HEIGHT = int(os.environ.get("CCNV_THUMBNAIL_H", "720"))

# ── Metadata / Publishing ─────────────────────────────────────────────────────

METADATA_LANGUAGE = os.environ.get("CCNV_META_LANG", "en")
METADATA_GENERATOR_TAG = "CC-Novel2Video v2"

# ── Analytics ─────────────────────────────────────────────────────────────────

ANALYTICS_ENABLED = os.environ.get("CCNV_ANALYTICS", "true").lower() != "false"

# ── Project Analyzer ─────────────────────────────────────────────────────────

ANALYZER_WORDS_PER_SEGMENT = int(os.environ.get("CCNV_WORDS_PER_SEG", "250"))
ANALYZER_WORDS_PER_MINUTE_TTS = int(os.environ.get("CCNV_WPM_TTS", "150"))
ANALYZER_SECONDS_PER_IMAGE = float(os.environ.get("CCNV_S_PER_IMG", "4.0"))
# FLUX on T4: ~4s per image at 1080p with 4 inference steps


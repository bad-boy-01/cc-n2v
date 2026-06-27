"""plugins/motion/ken_burns.py — MotionEngine Ken Burns plugin wrapper."""
from __future__ import annotations
from pathlib import Path
from plugins.base import MotionPlugin


class Plugin(MotionPlugin):
    def render(
        self, image_path, output_path, motion, duration_s, fps=24, resolution="1080p"
    ) -> Path:
        from lib.motion_engine import MotionEngine
        engine = MotionEngine.__new__(MotionEngine)
        engine.fps = fps
        engine.resolution = resolution
        return engine._render_clip(image_path, output_path, motion, duration_s)

    @property
    def plugin_id(self) -> str:
        return "ken_burns"

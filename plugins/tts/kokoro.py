"""plugins/tts/kokoro.py — Kokoro TTS plugin wrapper."""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
from plugins.base import TTSPlugin


class Plugin(TTSPlugin):
    def __init__(self):
        self._tts = None

    def load(self) -> None:
        pass  # KokoroTTS lazy-loads on first synthesis call

    def synthesize(
        self, text, output_path, voice="af_heart", speed=1.0, language="en"
    ) -> Tuple[Path, float]:
        if self._tts is None:
            from lib.kokoro_tts import KokoroTTS
            self._tts = KokoroTTS.__new__(KokoroTTS)
        return self._tts.synthesize_text(
            text, output_path=output_path, voice=voice, speed=speed
        )

    def unload(self) -> None:
        if self._tts:
            try:
                self._tts.unload_model()
            except Exception:
                pass
            self._tts = None

    @property
    def plugin_id(self) -> str:
        return "kokoro"

    @property
    def available_voices(self) -> List[str]:
        return ["af_heart", "af_bella", "am_adam", "am_michael", "bf_emma", "bm_george"]

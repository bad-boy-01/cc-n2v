"""plugins/tts/edge_tts.py — Microsoft Edge TTS plugin (cloud, free, no API key)."""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from plugins.base import TTSPlugin


class Plugin(TTSPlugin):
    """Uses edge-tts. Requires internet and: pip install edge-tts"""

    def load(self) -> None:
        try:
            import edge_tts  # noqa: F401
        except ImportError:
            raise ImportError("Install edge-tts: pip install edge-tts")

    def synthesize(
        self, text, output_path, voice="en-US-JennyNeural", speed=1.0, language="en"
    ) -> Tuple[Path, float]:
        import asyncio
        import edge_tts

        output_path = Path(output_path)

        async def _synth():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))

        asyncio.run(_synth())

        duration = 0.0
        if output_path.exists():
            try:
                from pydub.utils import mediainfo
                info = mediainfo(str(output_path))
                duration = float(info.get("duration", 0))
            except Exception:
                duration = len(text.split()) / 150.0

        return output_path, duration

    def unload(self) -> None:
        pass  # cloud service

    @property
    def plugin_id(self) -> str:
        return "edge-tts"

    @property
    def available_voices(self) -> List[str]:
        return ["en-US-JennyNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"]

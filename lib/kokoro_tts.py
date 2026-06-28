"""
lib/kokoro_tts.py — Kokoro TTS audio synthesis for CC-Novel2Video.

Primary:  Kokoro-82M (local, free, ~400MB, CPU-capable)
Fallback: edge-tts (Microsoft TTS via free API, cloud-based but no key needed)

Pipeline:
  projects/{name}/drafts/episode_N/narration/{sid}.txt
       ↓
  KokoroTTS.synthesize_episode(episode)
       ↓
  projects/{name}/audio/scene_{sid}.wav
  projects/{name}/audio/{sid}.duration   ← float seconds, used by VideoComposer

Scene-level checkpointing: skips already-generated audio files.
Staged loading: call unload_model() to free memory before Stage 4.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from lib.config import DEFAULT_VOICE, DEFAULT_TTS_SPEED

logger = logging.getLogger(__name__)

# ── Model singleton ───────────────────────────────────────────────────────────

_KOKORO_PIPELINE = None
_KOKORO_LOCK = None


def _get_lock():
    global _KOKORO_LOCK
    if _KOKORO_LOCK is None:
        import threading
        _KOKORO_LOCK = threading.Lock()
    return _KOKORO_LOCK


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _get_audio_duration(wav_path: Path) -> float:
    """Return duration in seconds of a .wav file without loading full audio."""
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        return info.duration
    except Exception:
        pass
    # Fallback: read WAV header
    try:
        import wave
        with wave.open(str(wav_path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 4.0  # safe default


def _write_duration(wav_path: Path, duration: float) -> None:
    dur_path = wav_path.parent / f"{wav_path.stem}.duration"
    dur_path.write_text(f"{duration:.4f}", encoding="utf-8")


# ── Fallback TTS: edge-tts ────────────────────────────────────────────────────

async def _edge_tts_synthesize(text: str, out_path: Path, voice: str = "en-US-AriaNeural") -> None:
    """edge-tts fallback — runs async, converts voice name to edge-tts format."""
    try:
        import edge_tts
        # Map Kokoro voice keys to edge-tts voices
        EDGE_VOICE_MAP = {
            "af_heart": "en-US-AriaNeural",
            "af_bella": "en-US-JennyNeural",
            "am_adam":  "en-US-GuyNeural",
            "am_michael": "en-US-ChristopherNeural",
        }
        # If the voice is not in the map (e.g. explicitly passed 'en-US-AndrewNeural'), use it directly
        edge_voice = EDGE_VOICE_MAP.get(voice, voice)
        communicate = edge_tts.Communicate(text, edge_voice)
        await communicate.save(str(out_path))
    except Exception as e:
        raise RuntimeError(f"edge-tts fallback failed: {e}") from e


def _edge_tts_sync(text: str, out_path: Path, voice: str) -> None:
    """Synchronous wrapper around edge-tts async call."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(_edge_tts_synthesize(text, out_path, voice))
    except RuntimeError:
        # No event loop in this thread
        asyncio.run(_edge_tts_synthesize(text, out_path, voice))


# ── Main TTS class ────────────────────────────────────────────────────────────

class KokoroTTS:
    """
    Kokoro TTS wrapper for episode-level audio synthesis.

    Usage (staged loading pattern)
    --------------------------------
    tts = KokoroTTS(project_name="my_novel")
    result = tts.synthesize_episode(episode=1)
    tts.unload_model()   # ← free memory before Stage 4
    """

    def __init__(
        self,
        project_name: str,
        projects_root: str = "projects",
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_TTS_SPEED,
        lang: str = "a",           # "a" = American English, "b" = British, "j" = Japanese
    ):
        self.project_name = project_name
        self.project_dir = Path(projects_root) / project_name
        self.voice = voice
        self.speed = speed
        self.lang = lang

        self.audio_dir = self.project_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline = None   # lazy-loaded
        self._backend: Optional[str] = None

    # ── Model management ──────────────────────────────────────────────────────

    def _load_kokoro(self) -> bool:
        """Load Kokoro pipeline. Returns True on success, False if unavailable."""
        global _KOKORO_PIPELINE
        with _get_lock():
            if _KOKORO_PIPELINE is not None:
                self._pipeline = _KOKORO_PIPELINE
                self._backend = "kokoro"
                return True
            try:
                from kokoro import KPipeline
                logger.info("Loading Kokoro TTS pipeline …")
                pipeline = KPipeline(lang_code=self.lang)
                _KOKORO_PIPELINE = pipeline
                self._pipeline = pipeline
                self._backend = "kokoro"
                logger.info("Kokoro TTS loaded ✅")
                return True
            except ImportError:
                logger.warning("kokoro package not installed — will use edge-tts fallback")
                return False
            except Exception as e:
                logger.warning(f"Kokoro load failed: {e} — will use edge-tts fallback")
                return False

    def unload_model(self) -> None:
        """Release Kokoro pipeline from memory."""
        global _KOKORO_PIPELINE
        with _get_lock():
            if _KOKORO_PIPELINE is not None:
                try:
                    del _KOKORO_PIPELINE
                except Exception:
                    pass
                _KOKORO_PIPELINE = None
        self._pipeline = None
        self._backend = None
        gc.collect()
        logger.info("KokoroTTS: model unloaded")

    # ── Single segment synthesis ──────────────────────────────────────────────

    def synthesize_segment(self, segment_id: str, text: str) -> Path:
        """
        Synthesize one segment. Returns path to .wav file.

        Tries Kokoro first, falls back to edge-tts.
        Skips if output .wav already exists.
        """
        out_path = self.audio_dir / f"scene_{segment_id}.wav"
        dur_path = self.audio_dir / f"{segment_id}.duration"

        if out_path.exists() and dur_path.exists():
            logger.debug(f"  {segment_id}: audio exists, skipping")
            return out_path

        text = text.strip()
        if not text:
            logger.warning(f"  {segment_id}: empty text, generating silence")
            self._write_silence(out_path, duration=3.0)
            _write_duration(out_path, 3.0)
            return out_path

        # Try Kokoro if the voice doesn't look like an edge-tts specific voice
        is_edge_voice = "-" in self.voice
        if not is_edge_voice and (self._backend == "kokoro" or self._load_kokoro()):
            try:
                self._kokoro_synthesize(text, out_path)
                duration = _get_audio_duration(out_path)
                _write_duration(out_path, duration)
                logger.info(f"  {segment_id}: ✅ kokoro ({duration:.1f}s)")
                return out_path
            except Exception as e:
                logger.warning(f"  {segment_id}: Kokoro failed ({e}), trying edge-tts")

        # Fallback: edge-tts
        try:
            _edge_tts_sync(text, out_path, self.voice)
            duration = _get_audio_duration(out_path)
            _write_duration(out_path, duration)
            self._backend = "edge-tts"
            logger.info(f"  {segment_id}: ✅ edge-tts ({duration:.1f}s)")
            return out_path
        except Exception as e:
            logger.error(f"  {segment_id}: All TTS backends failed: {e}")
            # Write silence so pipeline doesn't break
            self._write_silence(out_path, duration=4.0)
            _write_duration(out_path, 4.0)
            return out_path

    def _kokoro_synthesize(self, text: str, out_path: Path) -> None:
        """Run Kokoro inference and save to WAV."""
        import numpy as np

        generator = self._pipeline(text, voice=self.voice, speed=self.speed)
        audio_chunks = []
        sample_rate = 24000  # Kokoro default

        for _, _, audio in generator:
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            raise RuntimeError("Kokoro returned no audio")

        # Concatenate and save
        full_audio = np.concatenate(audio_chunks)
        self._save_wav(full_audio, out_path, sample_rate)

    def _save_wav(self, audio_array, out_path: Path, sample_rate: int = 24000) -> None:
        """Save numpy float32 audio array to WAV."""
        try:
            import soundfile as sf
            sf.write(str(out_path), audio_array, sample_rate)
        except ImportError:
            import wave
            import struct
            # Manual WAV write as last resort
            audio_int16 = (audio_array * 32767).astype("int16")
            with wave.open(str(out_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(audio_int16.tobytes())

    def _write_silence(self, out_path: Path, duration: float = 3.0) -> None:
        """Write a silent WAV file (used as placeholder on TTS failure)."""
        try:
            import numpy as np
            sample_rate = 24000
            samples = int(sample_rate * duration)
            silence = np.zeros(samples, dtype="float32")
            self._save_wav(silence, out_path, sample_rate)
        except Exception:
            # Ultra-minimal WAV header fallback
            out_path.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80\xbb\x00\x00\x00\x77\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")

    # ── Episode-level synthesis ───────────────────────────────────────────────

    def synthesize_episode(self, episode: int) -> Dict[str, Any]:
        """
        Synthesize audio for all segments in an episode.

        Reads narration text from:
            projects/{name}/drafts/episode_{N}/narration/{sid}.txt

        Falls back to:
            projects/{name}/scripts/episode_{N}.json → novel_text field

        Returns
        -------
        dict: audio_dir, segment_count, durations {segment_id: seconds}
        """
        # Load narration texts
        text_map = self._load_narration_texts(episode)
        if not text_map:
            logger.warning(f"No narration texts found for episode {episode}")
            return {"audio_dir": str(self.audio_dir), "segment_count": 0, "durations": {}}

        total = len(text_map)
        logger.info(f"[KokoroTTS] Episode {episode}: synthesizing {total} segments")

        if not self._load_kokoro():
            logger.info("[KokoroTTS] Using edge-tts fallback for all segments")

        durations: Dict[str, float] = {}
        succeeded = 0
        skipped = 0

        for sid, text in text_map.items():
            dur_path = self.audio_dir / f"{sid}.duration"

            # Check if already done
            out_path = self.audio_dir / f"scene_{sid}.wav"
            if out_path.exists() and dur_path.exists():
                try:
                    durations[sid] = float(dur_path.read_text().strip())
                    skipped += 1
                    continue
                except Exception:
                    pass

            wav_path = self.synthesize_segment(sid, text)
            dur_path2 = self.audio_dir / f"{sid}.duration"
            if dur_path2.exists():
                try:
                    durations[sid] = float(dur_path2.read_text().strip())
                except Exception:
                    durations[sid] = 4.0
            succeeded += 1

        # Write episode durations summary
        summary_path = self.audio_dir / f"episode_{episode}_durations.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(durations, f, ensure_ascii=False, indent=2)

        total_secs = sum(durations.values())
        logger.info(
            f"[KokoroTTS] Episode {episode} done: {succeeded} new, {skipped} skipped, "
            f"~{total_secs / 60:.1f} min total audio"
        )

        return {
            "audio_dir": str(self.audio_dir),
            "segment_count": total,
            "durations": durations,
            "total_seconds": total_secs,
            "succeeded": succeeded,
            "skipped": skipped,
        }

    def _load_narration_texts(self, episode: int) -> Dict[str, str]:
        """
        Load per-segment narration text.

        Primary:  projects/{name}/drafts/episode_{N}/narration/{sid}.txt
        Fallback: projects/{name}/scripts/episode_{N}.json → segments[].novel_text
        """
        text_map: Dict[str, str] = {}

        narr_dir = self.project_dir / "drafts" / f"episode_{episode}" / "narration"
        if narr_dir.exists():
            for txt_file in sorted(narr_dir.glob("*.txt")):
                sid = txt_file.stem
                text = txt_file.read_text(encoding="utf-8").strip()
                if text:
                    text_map[sid] = text

        if text_map:
            return text_map

        # Fallback: read directly from storyboard JSON
        script_path = self.project_dir / "scripts" / f"episode_{episode}.json"
        if script_path.exists():
            try:
                with open(script_path, encoding="utf-8") as f:
                    script = json.load(f)
                for seg in script.get("segments", []):
                    sid = seg.get("segment_id", "")
                    text = seg.get("novel_text", "").strip()
                    if sid and text:
                        text_map[sid] = text
            except Exception as e:
                logger.warning(f"Could not load script fallback: {e}")

        return text_map

    @property
    def backend(self) -> Optional[str]:
        return self._backend

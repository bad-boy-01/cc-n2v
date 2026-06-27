"""
lib/subtitle_generator.py — SRT and ASS subtitle generation for CC-Novel2Video.

Input:
  - projects/{name}/scripts/episode_{N}.json  (segment text + segment IDs)
  - projects/{name}/audio/{sid}.duration      (float seconds per segment)

Output:
  - projects/{name}/subtitles/episode_{N}.srt   (standard subtitle format)
  - projects/{name}/subtitles/episode_{N}.ass   (advanced styled subtitles)

The SubtitleGenerator is CPU-only and has no model dependencies.
Called during Stage 4 before FFmpeg composition.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lib.config import (
    DEFAULT_SUBTITLE_FONT,
    DEFAULT_SUBTITLE_FONT_SIZE,
    DEFAULT_SUBTITLE_COLOR,
    DEFAULT_SUBTITLE_OUTLINE,
)

logger = logging.getLogger(__name__)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class SubtitleEntry:
    index: int
    segment_id: str
    text: str
    start_seconds: float
    end_seconds: float

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


# ── SRT formatting ────────────────────────────────────────────────────────────

def _seconds_to_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timestamp HH:MM:SS,mmm"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds % 1) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert float seconds to ASS timestamp H:MM:SS.cc"""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds % 1) * 100))  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _wrap_text(text: str, max_chars: int = 45) -> str:
    """Wrap subtitle text at word boundaries."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        if len(test) <= max_chars:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\\N".join(lines)  # SRT/ASS newline


# ── Main class ────────────────────────────────────────────────────────────────

class SubtitleGenerator:
    """
    Generates SRT and ASS subtitle files from storyboard episode data.

    Usage
    -----
    gen = SubtitleGenerator(project_name="my_novel")
    paths = gen.generate_episode_subtitles(episode=1)
    # paths = {"srt": Path(...), "ass": Path(...)}
    """

    def __init__(
        self,
        project_name: str,
        projects_root: str = "projects",
        font: str = DEFAULT_SUBTITLE_FONT,
        font_size: int = DEFAULT_SUBTITLE_FONT_SIZE,
        primary_color: str = DEFAULT_SUBTITLE_COLOR,
        outline_color: str = DEFAULT_SUBTITLE_OUTLINE,
    ):
        self.project_name = project_name
        self.project_dir = Path(projects_root) / project_name
        self.font = font
        self.font_size = font_size
        self.primary_color = primary_color
        self.outline_color = outline_color

        self.subtitles_dir = self.project_dir / "subtitles"
        self.subtitles_dir.mkdir(parents=True, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_episode_script(self, episode: int) -> List[Dict]:
        """Load segments from episode storyboard JSON."""
        script_path = self.project_dir / "scripts" / f"episode_{episode}.json"
        if not script_path.exists():
            raise FileNotFoundError(f"Episode script not found: {script_path}")
        with open(script_path, encoding="utf-8") as f:
            script = json.load(f)
        return script.get("segments", [])

    def _load_audio_durations(self, episode: int) -> Dict[str, float]:
        """
        Load per-segment audio durations from .duration files.
        Falls back to episode_N_durations.json summary.
        """
        audio_dir = self.project_dir / "audio"
        durations: Dict[str, float] = {}

        # Try summary file first
        summary = audio_dir / f"episode_{episode}_durations.json"
        if summary.exists():
            try:
                with open(summary, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Scan individual .duration files
        for dur_file in sorted(audio_dir.glob("*.duration")):
            sid = dur_file.stem
            try:
                durations[sid] = float(dur_file.read_text().strip())
            except Exception:
                pass

        return durations

    # ── Entry point ───────────────────────────────────────────────────────────

    def generate_episode_subtitles(
        self,
        episode: int,
        overwrite: bool = False,
    ) -> Dict[str, Path]:
        """
        Generate SRT and ASS subtitle files for an episode.

        Returns
        -------
        dict with keys "srt" and "ass" → absolute Paths
        """
        srt_path = self.subtitles_dir / f"episode_{episode}.srt"
        ass_path = self.subtitles_dir / f"episode_{episode}.ass"

        if srt_path.exists() and ass_path.exists() and not overwrite:
            logger.info(f"[SubtitleGenerator] Episode {episode}: subtitles exist, skipping")
            return {"srt": srt_path, "ass": ass_path}

        segments = self._load_episode_script(episode)
        durations = self._load_audio_durations(episode)

        entries = self._build_entries(segments, durations)

        srt_content = self.to_srt(entries)
        ass_content = self.to_ass(entries)

        srt_path.write_text(srt_content, encoding="utf-8")
        ass_path.write_text(ass_content, encoding="utf-8")

        total_secs = entries[-1].end_seconds if entries else 0
        logger.info(
            f"[SubtitleGenerator] Episode {episode}: {len(entries)} entries, "
            f"~{total_secs / 60:.1f} min → {srt_path.name}, {ass_path.name}"
        )

        return {"srt": srt_path, "ass": ass_path}

    def _build_entries(
        self,
        segments: List[Dict],
        durations: Dict[str, float],
    ) -> List[SubtitleEntry]:
        """Build subtitle entries with cumulative timestamps."""
        entries = []
        cursor = 0.0

        for idx, seg in enumerate(segments, start=1):
            sid = seg.get("segment_id", f"seg_{idx}")
            text = seg.get("novel_text", "").strip()

            if not text:
                continue

            duration = durations.get(sid, float(seg.get("duration_seconds", 4)))
            duration = max(duration, 1.0)  # never less than 1 second

            # Split long segments at sentence boundaries for readability
            sub_texts = self._split_for_display(text, duration)

            sub_dur = duration / len(sub_texts)
            for sub_text in sub_texts:
                entries.append(SubtitleEntry(
                    index=len(entries) + 1,
                    segment_id=sid,
                    text=sub_text,
                    start_seconds=cursor,
                    end_seconds=cursor + sub_dur,
                ))
                cursor += sub_dur

        # Renumber
        for i, e in enumerate(entries, start=1):
            e.index = i

        return entries

    def _split_for_display(self, text: str, duration: float) -> List[str]:
        """
        Split long text into display-friendly chunks.
        Aim for ~42 chars per chunk or ~5 seconds per chunk.
        """
        # If text and duration are both short, keep as one
        if len(text) <= 80 or duration <= 5.0:
            return [text]

        import re
        # Split at sentence boundaries
        sentences = re.split(r"(?<=[.!?。！？])\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return [text]

        # Group sentences into chunks of roughly equal duration
        chunks = []
        current = ""
        target_chars = max(40, len(text) // max(1, int(duration / 5)))

        for sent in sentences:
            trial = (current + " " + sent).strip() if current else sent
            if current and len(trial) > target_chars:
                chunks.append(current)
                current = sent
            else:
                current = trial

        if current:
            chunks.append(current)

        return chunks if chunks else [text]

    # ── SRT output ────────────────────────────────────────────────────────────

    def to_srt(self, entries: List[SubtitleEntry]) -> str:
        """Generate SRT format string."""
        blocks = []
        for entry in entries:
            start = _seconds_to_srt_time(entry.start_seconds)
            end = _seconds_to_srt_time(entry.end_seconds)
            text = entry.text.replace("\\N", "\n")
            blocks.append(f"{entry.index}\n{start} --> {end}\n{text}\n")
        return "\n".join(blocks)

    # ── ASS output ────────────────────────────────────────────────────────────

    def to_ass(self, entries: List[SubtitleEntry]) -> str:
        """Generate ASS format string with styled subtitles."""
        header = self._ass_header()
        dialogue_lines = []

        for entry in entries:
            start = _seconds_to_ass_time(entry.start_seconds)
            end = _seconds_to_ass_time(entry.end_seconds)
            text = _wrap_text(entry.text, max_chars=42)
            dialogue_lines.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
            )

        return header + "\n".join(dialogue_lines) + "\n"

    def _ass_header(self) -> str:
        """Generate ASS file header with style definition."""
        return (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1920\n"
            "PlayResY: 1080\n"
            "ScaledBorderAndShadow: yes\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Default,{self.font},{self.font_size},"
            f"{self.primary_color},&H000000FF,"
            f"{self.outline_color},&H80000000,"
            f"0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

    # ── FFmpeg burned subtitle helper ─────────────────────────────────────────

    def get_ffmpeg_subtitle_filter(self, episode: int) -> str:
        """
        Return an FFmpeg filter string for burning subtitles into video.
        Used by VideoComposer when include_subtitles=True.
        """
        ass_path = self.subtitles_dir / f"episode_{episode}.ass"
        if not ass_path.exists():
            raise FileNotFoundError(f"ASS subtitle file not found: {ass_path}")
        # Escape backslashes for FFmpeg filter on Windows
        safe_path = str(ass_path).replace("\\", "/").replace(":", "\\:")
        return f"ass='{safe_path}'"

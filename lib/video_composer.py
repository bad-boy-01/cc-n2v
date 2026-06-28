"""
lib/video_composer.py — FFmpeg-based final video assembly for CC-Novel2Video.

Motion engine already outputs per-scene .mp4 clips (Ken Burns / pan effects).
This composer assembles them into the final episode video.

Pipeline (Stage 4 — CPU only, no model loading):

  cache/motion/scene_{id}.mp4   (from MotionEngine)
  +
  audio/scene_{id}.wav          (from KokoroTTS)
  +
  subtitles/episode_{N}.ass     (from SubtitleGenerator, optional)
  +
  bgm.mp3                       (optional background music)
  =
  output/episode_{N}.mp4        (final deliverable)

Strategy:
  1. Mux each motion clip with its matching audio (per-scene)
  2. Concatenate all muxed scene clips via FFmpeg concat demuxer
  3. Optionally burn subtitles
  4. Optionally mix in background music

All stages are resumable via file existence checks.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.config import DEFAULT_RESOLUTION, DEFAULT_VIDEO_CRF, DEFAULT_VIDEO_PRESET, BGM_VOLUME

logger = logging.getLogger(__name__)


# ── FFmpeg helpers ────────────────────────────────────────────────────────────

def _ffmpeg(*args, check: bool = True) -> subprocess.CompletedProcess:
    """Run FFmpeg with the given arguments. Raises on failure if check=True."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + list(args)
    logger.debug(f"FFmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (code {result.returncode}):\n{result.stderr[-1000:]}"
        )
    return result


def _ffmpeg_available() -> bool:
    """Check if FFmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, check=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 4.0


# ── Resolution map (height, width) ────────────────────────────────────────────

RESOLUTION_MAP = {
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k":    (3840, 2160),
    "16:9":  (1920, 1080),
}


class VideoComposer:
    """
    Assembles the final episode MP4 from per-scene motion clips and audio.

    Stage 4 of the pipeline — CPU only, no model loading.
    All intermediate files are cached; resume is supported.
    """

    def __init__(
        self,
        project_name: str,
        projects_root: str = "projects",
        resolution: str = DEFAULT_RESOLUTION,
        fps: int = 24,
        crf: int = DEFAULT_VIDEO_CRF,
        preset: str = DEFAULT_VIDEO_PRESET,
    ):
        self.project_name = project_name
        self.project_dir = Path(projects_root) / project_name
        self.resolution = resolution
        self.fps = fps
        self.crf = crf
        self.preset = preset

        w, h = RESOLUTION_MAP.get(resolution, (1920, 1080))
        self.width = w
        self.height = h

        # Directory layout
        self.motion_dir = self.project_dir / "cache" / "motion"
        self.audio_dir = self.project_dir / "audio"
        self.muxed_dir = self.project_dir / "cache" / "muxed"
        self.output_dir = self.project_dir / "output"
        self.subtitles_dir = self.project_dir / "subtitles"

        for d in [self.muxed_dir, self.output_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if not _ffmpeg_available():
            logger.warning("FFmpeg not found in PATH — video composition will fail!")

    # ── Public entry point ────────────────────────────────────────────────────

    def compose_episode(
        self,
        episode: int,
        include_subtitles: bool = True,
        bgm_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        """
        Compose the final episode video.

        Returns path to the output MP4 file.
        """
        output_path = self.output_dir / f"episode_{episode}.mp4"

        if output_path.exists() and not overwrite:
            logger.info(f"[VideoComposer] Episode {episode} output exists — skipping")
            return output_path

        logger.info(f"[VideoComposer] === Composing episode {episode} ===")

        segments = self._load_segments(episode)
        if not segments:
            raise ValueError(f"No segments found for episode {episode}")

        # Step 1: Mux each motion clip with its audio
        muxed_paths = self._mux_all_scenes(segments)

        if not muxed_paths:
            logger.error("[VideoComposer] No muxed clips produced — all image generation failed. Exiting gracefully.")
            return None

        # Step 2: Concatenate all muxed clips
        logger.info(f"[VideoComposer] Concatenating {len(muxed_paths)} scenes …")
        raw_concat = self.output_dir / f"episode_{episode}_raw.mp4"
        self._concat_clips(muxed_paths, raw_concat)

        # Step 3: Burn subtitles (optional)
        if include_subtitles:
            ass_path = self.subtitles_dir / f"episode_{episode}.ass"
            if ass_path.exists():
                logger.info("[VideoComposer] Burning subtitles …")
                subtitled = self.output_dir / f"episode_{episode}_subtitled.mp4"
                self._burn_subtitles(raw_concat, ass_path, subtitled)
                current = subtitled
            else:
                logger.warning(f"[VideoComposer] ASS subtitle file not found: {ass_path} — skipping")
                current = raw_concat
        else:
            current = raw_concat

        # Step 4: Mix background music (optional)
        if bgm_path and Path(bgm_path).exists():
            logger.info("[VideoComposer] Mixing background music …")
            with_bgm = self.output_dir / f"episode_{episode}_bgm.mp4"
            self._mix_bgm(current, Path(bgm_path), with_bgm)
            current = with_bgm

        # Step 5: Move final to output path
        if current != output_path:
            shutil.move(str(current), str(output_path))

        # Cleanup intermediates
        self._cleanup_intermediates(episode, output_path)

        duration = _get_video_duration(output_path)
        logger.info(
            f"[VideoComposer] ✅ Episode {episode} complete: {output_path.name} "
            f"({duration / 60:.1f} min)"
        )
        return output_path

    # ── Step 1: Per-scene muxing ──────────────────────────────────────────────

    def _mux_all_scenes(self, segments: List[Dict]) -> List[Path]:
        """Mux each motion clip with its audio. Returns ordered list of muxed paths."""
        muxed = []

        for seg in segments:
            sid = seg.get("segment_id", "")
            if not sid:
                continue

            muxed_path = self.muxed_dir / f"scene_{sid}.mp4"

            if muxed_path.exists():
                muxed.append(muxed_path)
                logger.debug(f"  {sid}: muxed clip exists, skipping")
                continue

            motion_clip = self.motion_dir / f"scene_{sid}.mp4"
            audio_clip = self.audio_dir / f"scene_{sid}.wav"

            if not motion_clip.exists():
                logger.warning(f"  {sid}: motion clip missing — skipping scene")
                continue

            if audio_clip.exists():
                self._mux_video_audio(motion_clip, audio_clip, muxed_path)
            else:
                # No audio: just copy motion clip as-is
                logger.warning(f"  {sid}: no audio clip — using silent motion clip")
                shutil.copy2(motion_clip, muxed_path)

            if muxed_path.exists():
                muxed.append(muxed_path)
                logger.info(f"  {sid}: ✅ muxed")
            else:
                logger.error(f"  {sid}: ❌ mux failed")

        return muxed

    def _mux_video_audio(self, video: Path, audio: Path, out: Path) -> None:
        """Combine video clip with audio, trimming/padding audio to match video duration."""
        video_dur = _get_video_duration(video)
        _ffmpeg(
            "-i", str(video),
            "-i", str(audio),
            "-c:v", "copy",            # no re-encode for video
            "-c:a", "aac",
            "-b:a", "128k",
            "-t", str(video_dur),      # trim to video length
            "-af", f"apad=whole_dur={video_dur}",  # pad audio if shorter
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            str(out),
        )

    # ── Step 2: Concatenation ─────────────────────────────────────────────────

    def _concat_clips(self, clip_paths: List[Path], out: Path) -> None:
        """
        Concatenate clips using FFmpeg concat demuxer (no re-encode of video stream).
        Writes a temporary concat list file.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            for p in clip_paths:
                # FFmpeg requires forward slashes and escaped spaces
                safe = str(p.resolve()).replace("\\", "/")
                f.write(f"file '{safe}'\n")
            list_file = Path(f.name)

        try:
            _ffmpeg(
                "-f", "concat",
                "-safe", "0",
                "-i", str(list_file),
                "-c:v", "libx264",
                "-crf", str(self.crf),
                "-preset", self.preset,
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",  # web-optimized
                str(out),
            )
        finally:
            list_file.unlink(missing_ok=True)

    # ── Step 3: Subtitle burning ──────────────────────────────────────────────

    def _burn_subtitles(self, video: Path, ass_path: Path, out: Path) -> None:
        """
        Burn ASS subtitles into video using FFmpeg subtitles filter.
        Re-encodes video — necessary for burned subtitles.
        """
        safe_ass = str(ass_path.resolve()).replace("\\", "/")
        # On Windows, colons in drive letters need escaping
        if len(safe_ass) > 1 and safe_ass[1] == ":":
            safe_ass = safe_ass[0] + "\\:" + safe_ass[2:]

        _ffmpeg(
            "-i", str(video),
            "-vf", f"ass='{safe_ass}'",
            "-c:v", "libx264",
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-c:a", "copy",
            str(out),
        )

    # ── Step 4: BGM mixing ────────────────────────────────────────────────────

    def _mix_bgm(self, video: Path, bgm: Path, out: Path) -> None:
        """
        Mix background music at reduced volume with existing narration audio.
        Loops BGM if shorter than video. Fades BGM out in last 3 seconds.
        """
        video_dur = _get_video_duration(video)
        bgm_filter = (
            f"[1:a]volume={BGM_VOLUME},"
            f"afade=t=out:st={max(0, video_dur - 3):.1f}:d=3,"
            f"aloop=loop=-1:size=2e+09[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first[aout]"
        )
        _ffmpeg(
            "-i", str(video),
            "-i", str(bgm),
            "-filter_complex", bgm_filter,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-t", str(video_dur),
            str(out),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_segments(self, episode: int) -> List[Dict]:
        """Load ordered segment list from episode storyboard JSON."""
        script_path = self.project_dir / "scripts" / f"episode_{episode}.json"
        if not script_path.exists():
            raise FileNotFoundError(f"Episode script not found: {script_path}")
        with open(script_path, encoding="utf-8") as f:
            script = json.load(f)
        return script.get("segments", [])

    def _cleanup_intermediates(self, episode: int, final_output: Path) -> None:
        """Remove intermediate files after successful composition."""
        if not final_output.exists():
            return
        for pattern in [f"episode_{episode}_raw.mp4", f"episode_{episode}_subtitled.mp4",
                         f"episode_{episode}_bgm.mp4"]:
            p = self.output_dir / pattern
            if p.exists() and p != final_output:
                try:
                    p.unlink()
                except Exception:
                    pass

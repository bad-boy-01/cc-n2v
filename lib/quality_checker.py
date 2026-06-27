"""
lib/quality_checker.py — Asset validation for all pipeline outputs.

Validates generated images, audio files, and video files.
Flags failures and provides structured results for auto-regeneration.

Called at the end of Stage 2 (images), Stage 3 (audio), Stage 4 (video).
"""

from __future__ import annotations

import logging
import struct
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_IMAGE_SIZE_KB = 5
MIN_AUDIO_SIZE_KB = 1
MIN_AUDIO_DURATION_S = 0.5
MAX_AUDIO_DURATION_S = 300.0
MIN_VIDEO_DURATION_S = 0.5
MIN_PIXEL_VARIANCE = 10  # below = black/white/blank image


@dataclass
class QualityResult:
    """Result of a single asset quality check."""
    path: str
    asset_type: str   # "image", "audio", "video"
    ok: bool
    severity: str     # "pass", "warn", "fail"
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EpisodeReport:
    """Quality report for an entire episode."""
    project: str
    episode: int
    images: List[QualityResult] = field(default_factory=list)
    audio: List[QualityResult] = field(default_factory=list)
    video: List[QualityResult] = field(default_factory=list)

    @property
    def images_passed(self) -> int:
        return sum(1 for r in self.images if r.ok)

    @property
    def audio_passed(self) -> int:
        return sum(1 for r in self.audio if r.ok)

    @property
    def failed_images(self) -> List[QualityResult]:
        return [r for r in self.images if not r.ok]

    @property
    def failed_audio(self) -> List[QualityResult]:
        return [r for r in self.audio if not r.ok]

    def summary(self) -> str:
        lines = [
            f"Quality Report — {self.project} Episode {self.episode}",
            f"  Images: {self.images_passed}/{len(self.images)} passed",
            f"  Audio:  {self.audio_passed}/{len(self.audio)} passed",
            f"  Video:  {len([r for r in self.video if r.ok])}/{len(self.video)} passed",
        ]
        for r in self.failed_images:
            lines.append(f"  ❌ Image {Path(r.path).name}: {', '.join(r.issues)}")
        for r in self.failed_audio:
            lines.append(f"  ❌ Audio {Path(r.path).name}: {', '.join(r.issues)}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "episode": self.episode,
            "images": [r.to_dict() for r in self.images],
            "audio": [r.to_dict() for r in self.audio],
            "video": [r.to_dict() for r in self.video],
            "summary": {
                "images_passed": self.images_passed,
                "images_total": len(self.images),
                "audio_passed": self.audio_passed,
                "audio_total": len(self.audio),
                "video_passed": len([r for r in self.video if r.ok]),
                "video_total": len(self.video),
            }
        }


class QualityChecker:
    """
    Asset quality validator.

    Usage
    -----
    checker = QualityChecker("projects")
    result = checker.check_image(Path("images/scene_001.png"))
    if not result.ok:
        print(result.issues)
    """

    def __init__(
        self,
        projects_root: Optional[str] = None,
        min_image_size_kb: int = MIN_IMAGE_SIZE_KB,
        min_audio_duration: float = MIN_AUDIO_DURATION_S,
    ):
        self.projects_root = Path(projects_root) if projects_root else Path("projects")
        self.min_image_size_kb = min_image_size_kb
        self.min_audio_duration = min_audio_duration

    # ── Image checks ──────────────────────────────────────────────────────────

    def check_image(self, path: Path) -> QualityResult:
        """
        Validate a generated image.

        Checks:
        - File exists
        - File size > min_image_size_kb
        - Valid PNG/JPEG header (not corrupt)
        - Pixel variance above threshold (not blank/black/white)
        """
        path = Path(path)
        issues: List[str] = []

        # Existence
        if not path.exists():
            return QualityResult(
                path=str(path), asset_type="image", ok=False,
                severity="fail", issues=["File does not exist"]
            )

        # File size
        size_kb = path.stat().st_size / 1024
        if size_kb < self.min_image_size_kb:
            issues.append(f"File too small ({size_kb:.1f}KB < {self.min_image_size_kb}KB)")

        # Header validity
        if not self._valid_image_header(path):
            issues.append("Invalid or corrupt image header")

        # Pixel variance check
        if not issues:  # only if header is valid
            variance = self._compute_pixel_variance(path)
            if variance is not None and variance < MIN_PIXEL_VARIANCE:
                issues.append(
                    f"Image appears blank (pixel variance={variance:.1f} < {MIN_PIXEL_VARIANCE})"
                )

        severity = "fail" if issues else "pass"
        return QualityResult(
            path=str(path), asset_type="image",
            ok=not bool(issues), severity=severity, issues=issues
        )

    def _valid_image_header(self, path: Path) -> bool:
        """Check PNG/JPEG magic bytes."""
        try:
            with open(path, "rb") as f:
                header = f.read(8)
            # PNG: \x89PNG\r\n\x1a\n
            if header[:4] == b"\x89PNG":
                return True
            # JPEG: \xFF\xD8\xFF
            if header[:3] == b"\xff\xd8\xff":
                return True
            # WebP: RIFF????WEBP
            if header[:4] == b"RIFF":
                return True
            return False
        except Exception:
            return False

    def _compute_pixel_variance(self, path: Path) -> Optional[float]:
        """Compute pixel variance to detect blank/monochrome images."""
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(str(path)).convert("L")  # grayscale
            arr = np.array(img, dtype=float)
            return float(arr.var())
        except Exception:
            return None

    # ── Audio checks ──────────────────────────────────────────────────────────

    def check_audio(
        self,
        path: Path,
        expected_duration: Optional[float] = None,
    ) -> QualityResult:
        """
        Validate a synthesized audio file.

        Checks:
        - File exists
        - File size > 1KB
        - Valid WAV header
        - Duration in reasonable range
        - Duration matches expected (if provided)
        """
        path = Path(path)
        issues: List[str] = []

        if not path.exists():
            return QualityResult(
                path=str(path), asset_type="audio", ok=False,
                severity="fail", issues=["File does not exist"]
            )

        size_kb = path.stat().st_size / 1024
        if size_kb < MIN_AUDIO_SIZE_KB:
            issues.append(f"Audio file too small ({size_kb:.1f}KB)")

        # WAV header check
        duration = self._get_wav_duration(path)
        if duration is None:
            issues.append("Invalid or corrupt WAV file")
        else:
            if duration < self.min_audio_duration:
                issues.append(
                    f"Audio too short ({duration:.2f}s < {self.min_audio_duration}s)"
                )
            elif duration > MAX_AUDIO_DURATION_S:
                issues.append(
                    f"Audio suspiciously long ({duration:.1f}s > {MAX_AUDIO_DURATION_S}s)"
                )
            if expected_duration and abs(duration - expected_duration) > 2.0:
                issues.append(
                    f"Duration mismatch: got {duration:.2f}s, expected {expected_duration:.2f}s"
                )

        severity = "fail" if issues else "pass"
        return QualityResult(
            path=str(path), asset_type="audio",
            ok=not bool(issues), severity=severity, issues=issues
        )

    def _get_wav_duration(self, path: Path) -> Optional[float]:
        """Parse WAV header to get duration."""
        try:
            with open(path, "rb") as f:
                riff = f.read(4)
                if riff != b"RIFF":
                    return None
                f.seek(22)  # NumChannels
                channels = struct.unpack("<H", f.read(2))[0]
                sample_rate = struct.unpack("<I", f.read(4))[0]
                f.seek(34)  # BitsPerSample
                bits_per_sample = struct.unpack("<H", f.read(2))[0]
                f.seek(40)  # DataChunkSize
                data_size = struct.unpack("<I", f.read(4))[0]
                if channels and sample_rate and bits_per_sample:
                    bytes_per_sample = bits_per_sample // 8
                    total_samples = data_size // (channels * bytes_per_sample)
                    return total_samples / sample_rate
        except Exception:
            pass
        return None

    # ── Video checks ──────────────────────────────────────────────────────────

    def check_video(self, path: Path) -> QualityResult:
        """
        Validate a rendered video file using ffprobe.

        Checks:
        - File exists
        - Valid MP4 container
        - Has video stream
        - Has audio stream
        - Duration > 0
        """
        path = Path(path)
        issues: List[str] = []

        if not path.exists():
            return QualityResult(
                path=str(path), asset_type="video", ok=False,
                severity="fail", issues=["File does not exist"]
            )

        if path.stat().st_size < 10_000:  # < 10KB is definitely corrupt
            issues.append("Video file too small (likely corrupt)")

        # Use ffprobe for detailed validation
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-print_format", "json",
                    "-show_streams", "-show_format",
                    str(path)
                ],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                issues.append("ffprobe failed — file may be corrupt")
            else:
                import json
                probe = json.loads(result.stdout)
                streams = probe.get("streams", [])
                codecs = {s.get("codec_type") for s in streams}

                if "video" not in codecs:
                    issues.append("No video stream found")
                if "audio" not in codecs:
                    issues.append("No audio stream found")

                duration = float(probe.get("format", {}).get("duration", 0))
                if duration < MIN_VIDEO_DURATION_S:
                    issues.append(f"Video duration too short ({duration:.2f}s)")

        except FileNotFoundError:
            # ffprobe not available — do basic size check only
            logger.debug("ffprobe not available — skipping detailed video validation")
        except Exception as e:
            issues.append(f"Video probe error: {e}")

        severity = "fail" if issues else "pass"
        return QualityResult(
            path=str(path), asset_type="video",
            ok=not bool(issues), severity=severity, issues=issues
        )

    # ── Episode-level validation ──────────────────────────────────────────────

    def validate_episode(
        self,
        project_name: str,
        episode: int,
    ) -> EpisodeReport:
        """
        Validate all assets for an episode.

        Returns an EpisodeReport with per-asset results.
        """
        project_dir = self.projects_root / project_name
        report = EpisodeReport(project=project_name, episode=episode)

        # Check images
        images_dir = project_dir / "images"
        if images_dir.exists():
            for img_path in sorted(images_dir.glob(f"*E{episode}S*.png")):
                report.images.append(self.check_image(img_path))
            for img_path in sorted(images_dir.glob(f"*E{episode:02d}S*.png")):
                if img_path not in [Path(r.path) for r in report.images]:
                    report.images.append(self.check_image(img_path))

        # Check audio
        audio_dir = project_dir / "audio"
        if audio_dir.exists():
            for wav_path in sorted(audio_dir.glob("*.wav")):
                # Load expected duration from sidecar if exists
                dur_file = wav_path.with_suffix(".duration")
                expected = None
                if dur_file.exists():
                    try:
                        expected = float(dur_file.read_text().strip())
                    except Exception:
                        pass
                report.audio.append(self.check_audio(wav_path, expected))

        # Check output video
        output_dir = project_dir / "output"
        if output_dir.exists():
            for mp4_path in output_dir.glob(f"episode_{episode}.mp4"):
                report.video.append(self.check_video(mp4_path))

        logger.info(f"\n{report.summary()}")
        return report

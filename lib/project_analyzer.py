"""
lib/project_analyzer.py — Pre-run feasibility analyzer.

Runs BEFORE any model loads. Pure Python, no GPU required.
Estimates resource usage and feasibility so users know what to expect
before committing hours of GPU time on Kaggle.

Usage
-----
analyzer = ProjectAnalyzer("projects")
report = analyzer.analyze("source/my_novel.txt", mode="novel")
print(report.summary())

# Or from CLI:
# python run_pipeline.py --analyze --input my_novel.txt
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Estimation constants ─────────────────────────────────────────────────────
# (overridable via lib/config.py)

WORDS_PER_SEGMENT = 250        # avg segment length (~4s narration at 150 WPM)
WORDS_PER_MINUTE_TTS = 150     # Kokoro TTS approximate speech rate
SECONDS_PER_IMAGE_FLUX = 4.0   # FLUX.1-schnell on T4, 1080p, 4 steps
SECONDS_PER_IMAGE_SDXL = 8.0   # SDXL on T4, 1080p, 30 steps

# Estimated VRAM usage peaks (GB)
VRAM_QWEN_4BIT = 5.0           # Qwen2.5-7B-Instruct 4-bit
VRAM_FLUX_SCHNELL = 7.5        # FLUX.1-schnell fp16
VRAM_SDXL = 5.5                # SDXL base
VRAM_KOKORO = 0.0              # Kokoro runs on CPU

# Storage estimates per asset
IMAGE_SIZE_MB = 2.0            # avg compressed PNG at 1080p
AUDIO_SIZE_MB_PER_MINUTE = 5.0
VIDEO_SIZE_MB_PER_MINUTE = 50.0

# Kaggle T4 total VRAM
KAGGLE_T4_VRAM = 15.0

# Feasibility thresholds
FEASIBILITY_WARN_HOURS = 4.0   # warn if > 4 GPU hours
FEASIBILITY_FAIL_HOURS = 12.0  # fail if > 12 GPU hours (Kaggle session limit)


@dataclass
class AnalysisReport:
    """Full analysis report for a project."""

    # Input
    input_file: str = ""
    mode: str = "novel"
    resolution: str = "1080p"

    # Text stats
    word_count: int = 0
    char_count: int = 0
    estimated_segments: int = 0
    estimated_episodes: int = 1

    # Content estimates
    estimated_images: int = 0
    estimated_audio_minutes: float = 0.0
    estimated_runtime_minutes: float = 0.0

    # Resource estimates
    estimated_vram_gb: float = 0.0
    estimated_storage_gb: float = 0.0
    estimated_gpu_hours_flux: float = 0.0
    estimated_gpu_hours_sdxl: float = 0.0
    estimated_gpu_hours_llm: float = 0.0
    estimated_total_gpu_hours: float = 0.0

    # Feasibility
    feasibility: str = "unknown"   # feasible | warn | infeasible
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendation: str = ""

    def summary(self) -> str:
        """Human-readable summary for CLI / WebUI."""
        icon = {"feasible": "✅", "warn": "⚠️", "infeasible": "❌"}.get(
            self.feasibility, "❓"
        )
        lines = [
            f"{icon} Project Analysis Report",
            f"",
            f"  Input:            {self.input_file}",
            f"  Mode:             {self.mode}",
            f"  Word count:       {self.word_count:,}",
            f"  Segments:         {self.estimated_segments:,}",
            f"  Episodes:         {self.estimated_episodes}",
            f"",
            f"  Estimated images: {self.estimated_images}",
            f"  Estimated audio:  {self.estimated_audio_minutes:.1f} min",
            f"  Estimated video:  {self.estimated_runtime_minutes:.1f} min",
            f"",
            f"  Peak VRAM:        {self.estimated_vram_gb:.1f} GB (T4 has {KAGGLE_T4_VRAM:.0f} GB)",
            f"  Storage:          {self.estimated_storage_gb:.1f} GB",
            f"  GPU time:         {self.estimated_total_gpu_hours:.1f} hours",
            f"  Feasibility:      {self.feasibility.upper()}",
        ]
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")
        if self.errors:
            lines.append("")
            for e in self.errors:
                lines.append(f"  ❌ {e}")
        if self.recommendation:
            lines.append(f"")
            lines.append(f"  💡 {self.recommendation}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ProjectAnalyzer:
    """
    Pre-run feasibility analyzer.

    Reads source text, estimates resource requirements, and produces
    an AnalysisReport BEFORE any model is loaded. No GPU, no model needed.
    """

    def __init__(
        self,
        projects_root: Optional[str] = None,
        words_per_segment: int = WORDS_PER_SEGMENT,
        wpm_tts: int = WORDS_PER_MINUTE_TTS,
        seconds_per_image: float = SECONDS_PER_IMAGE_FLUX,
    ):
        self.projects_root = Path(projects_root) if projects_root else Path("projects")
        self.words_per_segment = words_per_segment
        self.wpm_tts = wpm_tts
        self.seconds_per_image = seconds_per_image

        # Try to load config overrides
        try:
            from lib import config
            self.words_per_segment = getattr(config, "ANALYZER_WORDS_PER_SEGMENT", words_per_segment)
            self.wpm_tts = getattr(config, "ANALYZER_WORDS_PER_MINUTE_TTS", wpm_tts)
            self.seconds_per_image = getattr(config, "ANALYZER_SECONDS_PER_IMAGE", seconds_per_image)
        except Exception:
            pass

    def analyze(
        self,
        input_file: str,
        mode: str = "novel",
        resolution: str = "1080p",
        max_words_per_episode: int = 15_000,
        difficulty_easy_fraction: float = 0.5,
    ) -> AnalysisReport:
        """
        Analyze a source file and produce an AnalysisReport.

        Parameters
        ----------
        input_file : str
            Path to source text file (or directory for manhwa)
        mode : str
            Pipeline mode: "novel", "manhwa", "manhwa_panels"
        resolution : str
            Output resolution: "720p", "1080p", "4k"
        max_words_per_episode : int
            Word limit per episode for splitting (default 15k ≈ 22 min)
        difficulty_easy_fraction : float
            Fraction of scenes estimated as easy/medium (→ SDXL). Default 0.5.

        Returns
        -------
        AnalysisReport
        """
        report = AnalysisReport(
            input_file=input_file,
            mode=mode,
            resolution=resolution,
        )

        # ── 1. Read source content ────────────────────────────────────────────
        source_path = Path(input_file)
        if not source_path.exists():
            report.errors.append(f"Input file not found: {input_file}")
            report.feasibility = "infeasible"
            return report

        if source_path.is_dir():
            # Manhwa mode: count image files
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            images = [f for f in source_path.iterdir() if f.suffix.lower() in exts]
            report.estimated_images = len(images)
            report.estimated_segments = len(images)
            report.word_count = 0
            report.char_count = 0
        else:
            text = source_path.read_text(encoding="utf-8", errors="ignore")
            words = text.split()
            report.word_count = len(words)
            report.char_count = len(text)
            report.estimated_segments = max(1, math.ceil(report.word_count / self.words_per_segment))
            report.estimated_images = report.estimated_segments

        # ── 2. Episode split ──────────────────────────────────────────────────
        if report.word_count > 0 and max_words_per_episode > 0:
            report.estimated_episodes = max(
                1, math.ceil(report.word_count / max_words_per_episode)
            )
        else:
            report.estimated_episodes = 1

        # ── 3. Audio & runtime ────────────────────────────────────────────────
        if report.word_count > 0:
            report.estimated_audio_minutes = round(
                report.word_count / self.wpm_tts, 1
            )
        else:
            # Manhwa: estimate from image count (4s per panel)
            report.estimated_audio_minutes = round(report.estimated_images * 4 / 60, 1)

        report.estimated_runtime_minutes = report.estimated_audio_minutes

        # ── 4. VRAM peak ──────────────────────────────────────────────────────
        # Stages never overlap, so peak = max of individual stages
        report.estimated_vram_gb = max(
            VRAM_QWEN_4BIT,
            VRAM_FLUX_SCHNELL if not difficulty_easy_fraction == 1.0 else VRAM_SDXL,
        )

        # ── 5. Storage ────────────────────────────────────────────────────────
        image_gb = (report.estimated_images * IMAGE_SIZE_MB) / 1024
        audio_gb = (report.estimated_audio_minutes * AUDIO_SIZE_MB_PER_MINUTE) / 1024
        video_gb = (report.estimated_runtime_minutes * VIDEO_SIZE_MB_PER_MINUTE) / 1024
        report.estimated_storage_gb = round(image_gb + audio_gb + video_gb, 2)

        # ── 6. GPU time ───────────────────────────────────────────────────────
        n_flux = math.ceil(report.estimated_images * (1.0 - difficulty_easy_fraction))
        n_sdxl = report.estimated_images - n_flux

        report.estimated_gpu_hours_flux = round(
            (n_flux * SECONDS_PER_IMAGE_FLUX) / 3600, 2
        )
        report.estimated_gpu_hours_sdxl = round(
            (n_sdxl * SECONDS_PER_IMAGE_SDXL) / 3600, 2
        )
        # LLM: ~1 token/ms, ~200 tokens/segment, ~500ms per batch of 25
        llm_batches = math.ceil(report.estimated_segments / 25)
        report.estimated_gpu_hours_llm = round((llm_batches * 30) / 3600, 2)

        report.estimated_total_gpu_hours = round(
            report.estimated_gpu_hours_flux
            + report.estimated_gpu_hours_sdxl
            + report.estimated_gpu_hours_llm,
            2
        )

        # ── 7. Feasibility judgment ───────────────────────────────────────────
        report.warnings = []
        report.errors = []

        if report.estimated_vram_gb > KAGGLE_T4_VRAM:
            report.errors.append(
                f"Estimated VRAM {report.estimated_vram_gb:.1f}GB exceeds T4 limit "
                f"({KAGGLE_T4_VRAM:.0f}GB)"
            )

        if report.estimated_total_gpu_hours > FEASIBILITY_FAIL_HOURS:
            report.warnings.append(
                f"Estimated GPU time {report.estimated_total_gpu_hours:.1f}h exceeds "
                f"Kaggle session limit (~{FEASIBILITY_FAIL_HOURS:.0f}h). "
                f"Split into {report.estimated_episodes} episodes and run separately."
            )

        if report.estimated_total_gpu_hours > FEASIBILITY_WARN_HOURS:
            report.warnings.append(
                f"GPU time {report.estimated_total_gpu_hours:.1f}h is significant. "
                f"Use difficulty routing to reduce SDXL/FLUX usage."
            )

        if report.estimated_storage_gb > 5.0:
            report.warnings.append(
                f"Storage {report.estimated_storage_gb:.1f}GB may fill Kaggle disk quota. "
                f"Use --dry-run to verify before full run."
            )

        if resolution == "4k" and VRAM_FLUX_SCHNELL > KAGGLE_T4_VRAM * 0.8:
            report.warnings.append(
                "4K resolution may OOM on T4. Recommend --resolution 1080p."
            )

        # Set feasibility
        if report.errors:
            report.feasibility = "infeasible"
        elif report.warnings:
            report.feasibility = "warn"
        else:
            report.feasibility = "feasible"

        # ── 8. Recommendation ─────────────────────────────────────────────────
        recs = []
        if report.estimated_episodes > 1:
            recs.append(f"Run as {report.estimated_episodes} separate episodes")
        if difficulty_easy_fraction < 0.5:
            recs.append("Enable difficulty routing (--difficulty-routing) to use SDXL for easy scenes")
        if report.estimated_total_gpu_hours < 1.0:
            recs.append("Project is lightweight — consider using FLUX for all scenes for best quality")

        report.recommendation = ". ".join(recs) if recs else "Ready to generate."

        return report

    def analyze_project_dir(
        self,
        project_name: str,
        episode: int = 1,
    ) -> AnalysisReport:
        """Analyze an already-initialized project directory."""
        project_dir = self.projects_root / project_name
        source_dir = project_dir / "source"
        episode_file = source_dir / f"episode_{episode}.txt"

        if episode_file.exists():
            return self.analyze(str(episode_file))

        # Find any text file
        txt_files = list(source_dir.glob("*.txt")) if source_dir.exists() else []
        if txt_files:
            return self.analyze(str(txt_files[0]))

        report = AnalysisReport()
        report.errors.append(f"No source files found in {source_dir}")
        report.feasibility = "infeasible"
        return report

    def save_report(
        self,
        report: AnalysisReport,
        project_dir: Path,
    ) -> Path:
        """Save AnalysisReport to project_dir/analysis_report.json."""
        out = Path(project_dir) / "analysis_report.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        return out

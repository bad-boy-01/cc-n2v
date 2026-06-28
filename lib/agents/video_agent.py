"""
lib/agents/video_agent.py — Pipeline orchestrator with staged model loading.

CRITICAL DESIGN: Models are never loaded simultaneously.
Each stage unloads before the next stage loads.

Stage isolation:
  Stage 1 (LLM):    Load Qwen → story/character/storyboard → Unload Qwen → gc.collect() + empty_cache()
  Stage 2 (Image):  Load FLUX → generate images           → Unload FLUX → gc.collect() + empty_cache()
  Stage 3 (Audio):  Load Kokoro → synthesize audio        → Unload Kokoro
  Stage 4 (Compose): FFmpeg only (CPU, no model loading)

Resume support via project_state.json.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from lib.analytics import PipelineAnalytics
from lib.config import DEFAULT_LLM
from lib.model_manager import ModelManager

logger = logging.getLogger(__name__)


def _free_vram() -> None:
    """Force full GPU memory release between stages (legacy shim)."""
    ModelManager().free_vram()


class PipelineState:
    """
    Persistent resume manifest — project_state.json.

    Written after each stage completes so Kaggle session restarts
    can resume from the last completed stage.
    """

    FIELDS = [
        "storyboard_complete",
        "images_complete",
        "audio_complete",
        "motion_complete",
        "video_complete",
    ]

    def __init__(self, project_dir: Path):
        self.path = project_dir / "project_state.json"
        self._state = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        return {
            "storyboard_complete": False,
            "images_complete": False,
            "audio_complete": False,
            "motion_complete": False,
            "video_complete": False,
            "last_scene": None,
            "last_batch": None,
            "episode": None,
        }

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    def mark(self, field: str, value: Any = True) -> None:
        self._state[field] = value
        self.save()

    def get(self, field: str, default: Any = None) -> Any:
        return self._state.get(field, default)

    def __repr__(self) -> str:
        return json.dumps(self._state, indent=2)


class VideoAgent:
    """
    Top-level pipeline orchestrator.

    Runs all 4 stages with strict model isolation and full resume support.
    """

    def __init__(
        self,
        project_name: str,
        episode: int = 1,
        projects_root: Optional[str] = None,
        input_file: Optional[str] = None,
        mode: str = "novel",
        llm: str = DEFAULT_LLM,
        load_in_4bit: bool = True,
        resolution: str = "1080p",
        storyboard_only: bool = False,
        dry_run: bool = False,
        fast_cpu_overlap: bool = False,
        max_scenes: Optional[int] = None,
        style: str = "default",
    ):
        from lib.project_manager import ProjectManager
        self.project_name = project_name
        self.episode = episode
        self.pm = ProjectManager(projects_root)
        self.project_dir = self.pm.get_project_path(project_name)
        self.input_file = input_file
        self.mode = mode
        self.llm = llm
        self.load_in_4bit = load_in_4bit
        self.resolution = resolution
        self.storyboard_only = storyboard_only
        self.dry_run = dry_run
        self.fast_cpu_overlap = fast_cpu_overlap
        self.max_scenes = max_scenes
        self.state = PipelineState(self.project_dir)
        self.mm = ModelManager()   # centralized model lifecycle

        self._agent_kwargs = dict(
            projects_root=projects_root,
            llm=llm,
            load_in_4bit=load_in_4bit,
        )
        self.style = style
        self.analytics = PipelineAnalytics(self.project_name, projects_root)
        self.analytics.set_model_stack(llm=llm)

    def _log(self, msg: str) -> None:
        logger.info(f"[VideoAgent][{self.project_name}/ep{self.episode}] {msg}")

    # ── Stage 0: Pre-processing (Manhwa Modes) ────────────────────────────────
    
    def _stage0_manhwa_ocr(self) -> None:
        if self.state.get("manhwa_ocr_complete"):
            self._log("Stage 0 (OCR) already complete — skipping")
            return
            
        self._log("=== STAGE 0: Manhwa OCR ===")
        if not self.input_file or not Path(self.input_file).exists():
            raise FileNotFoundError(f"Manhwa OCR mode requires a valid input_file (directory of images). Got: {self.input_file}")
            
        from lib.manhwa_ocr import ManhwaOCR
        ocr = ManhwaOCR(
            project_name=self.project_name, 
            projects_root=str(self.project_dir.parent)
        )
        # Extract text and save to source/chapter_N_ocr.txt
        ocr.extract_folder(Path(self.input_file))
        ocr.unload()
        _free_vram()
        
        # Copy the extracted OCR text to be used as the novel text for this episode
        source_dir = self.project_dir / "source"
        ocr_file = source_dir / f"{Path(self.input_file).name}_ocr.txt"
        episode_file = source_dir / f"episode_{self.episode}.txt"
        
        if ocr_file.exists():
            import shutil
            shutil.copy2(ocr_file, episode_file)
            self._log(f"Copied OCR text to {episode_file.name}")
            
        self.state.mark("manhwa_ocr_complete", True)
        self._log("Stage 0 complete ✅")

    def _stage0_manhwa_panels(self) -> None:
        if self.state.get("manhwa_panels_complete"):
            self._log("Stage 0 (Panels) already complete — skipping")
            return
            
        self._log("=== STAGE 0: Manhwa Panel Detection ===")
        if not self.input_file or not Path(self.input_file).exists():
            raise FileNotFoundError(f"Manhwa Panels mode requires a valid input_file (directory of images). Got: {self.input_file}")
            
        from lib.manhwa_panel_detector import ManhwaPanelDetector
        detector = ManhwaPanelDetector(
            project_name=self.project_name, 
            projects_root=str(self.project_dir.parent)
        )
        
        # Sort images by name
        img_dir = Path(self.input_file)
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        images = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in valid_exts])
        
        if not images:
            raise ValueError(f"No valid images found in {img_dir}")
            
        panels = detector.process_chapter(images)
        
        # Create a synthetic storyboard script so MotionEngine + VideoComposer know what to do
        script_dir = self.project_dir / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        script_file = script_dir / f"episode_{self.episode}.json"
        
        segments = []
        for sid, path in panels:
            segments.append({
                "segment_id": sid,
                "novel_text": "",  # No narration text
                "camera_motion": "Pan Down" if "01" in sid else "Zoom In", # Basic random defaults
                "duration_seconds": 4.0, # Default duration per panel
                "generated_assets": {
                    "storyboard_image": str(path)
                }
            })
            
        script_data = {
            "episode": self.episode,
            "segments": segments
        }
        
        with open(script_file, "w", encoding="utf-8") as f:
            json.dump(script_data, f, ensure_ascii=False, indent=2)
            
        self.state.mark("manhwa_panels_complete", True)
        self._log(f"Stage 0 complete ✅ — Synthesized storyboard with {len(panels)} panels")

    # ── Stage 1: LLM (Qwen) ──────────────────────────────────────────────────

    def _stage1_storyboard(self) -> None:
        if self.state.get("storyboard_complete"):
            self._log("Stage 1 already complete — skipping")
            return

        self._log("=== STAGE 1: Storyboard Generation (LLM) ===")

        from lib.agents.story_agent import StoryAgent
        from lib.agents.planner_agent import PlannerAgent
        from lib.agents.director_agent import DirectorAgent
        from lib.agents.character_agent import CharacterAgent
        from lib.agents.clue_agent import ClueAgent
        from lib.agents.storyboard_agent import StoryboardAgent
        from lib.agents.narration_agent import NarrationAgent

        self.analytics.start_stage("storyboard")

        # Story segmentation
        story = StoryAgent(self.project_name, episode=self.episode, **self._agent_kwargs)
        story.run()

        # Episode planning (key vs filler scenes)
        planner = PlannerAgent(self.project_name, episode=self.episode, **self._agent_kwargs)
        planner.run()

        # Director agent (visual style & pacing)
        director = DirectorAgent(self.project_name, style=self.style, **self._agent_kwargs)
        director.run()

        # Character + clue extraction (share same Qwen instance via singleton)
        char = CharacterAgent(self.project_name, episode=self.episode, **self._agent_kwargs)
        char.run()

        clue = ClueAgent(self.project_name, episode=self.episode, **self._agent_kwargs)
        clue.run()

        # Storyboard generation in batches of 25
        sb = StoryboardAgent(self.project_name, episode=self.episode, **self._agent_kwargs)
        sb.run()

        # Narration text prep
        narr = NarrationAgent(self.project_name, episode=self.episode, **self._agent_kwargs)
        narr.run()

        # CRITICAL: unload Qwen before Stage 2
        self.mm.unload_llm(self.llm)

        self.analytics.end_stage("storyboard")
        self.state.mark("storyboard_complete", True)
        self._log("Stage 1 complete ✅")

    # ── Stage 2: Image Generation (FLUX) ─────────────────────────────────────

    def _stage2_images(self) -> None:
        if self.storyboard_only:
            self._log("--mode storyboard_only: stopping after Stage 1")
            return

        if self.state.get("images_complete"):
            self._log("Stage 2 already complete — skipping")
            return

        self._log("=== STAGE 2: Image Generation (FLUX) ===")

        self.analytics.start_stage("images")
        from lib.image_generator import ImageGenerator
        gen = ImageGenerator(
            project_name=self.project_name,
            projects_root=str(self.project_dir.parent),
            resolution=self.resolution,
            dry_run=self.dry_run,
        )
        gen.generate_episode(self.episode, max_scenes=self.max_scenes)

        # Unload FLUX before Stage 3 via ModelManager
        self.mm.unload_image(gen)

        self.analytics.end_stage("images")

        # Quality check images
        from lib.quality_checker import QualityChecker
        qc = QualityChecker(str(self.project_dir.parent))
        report = qc.validate_episode(self.project_name, self.episode)
        if report.failed_images:
            self._log(f"Warning: {len(report.failed_images)} images failed quality checks.")

        self.state.mark("images_complete", True)
        self._log("Stage 2 complete ✅")

    # ── Stage 3: Audio (Kokoro) ───────────────────────────────────────────────

    def _stage3_audio(self) -> None:
        if self.state.get("audio_complete"):
            self._log("Stage 3 already complete — skipping")
            return

        self._log("=== STAGE 3: Audio Synthesis (Kokoro TTS) ===")

        self.analytics.start_stage("audio")
        from lib.kokoro_tts import KokoroTTS
        tts = KokoroTTS(
            project_name=self.project_name,
            projects_root=str(self.project_dir.parent),
        )
        tts.synthesize_episode(self.episode)
        # Kokoro runs on CPU — ModelManager handles gc.collect()
        self.mm.unload_tts(tts)

        self.analytics.end_stage("audio")

        self.state.mark("audio_complete", True)
        self._log("Stage 3 complete ✅")

    # ── Stage 3b: Motion Generation (Ken Burns) ───────────────────────────────

    def _stage3b_motion(self) -> None:
        if self.state.get("motion_complete"):
            self._log("Stage 3b already complete — skipping")
            return

        self._log("=== STAGE 3b: Motion Generation ===")

        self.analytics.start_stage("motion")
        from lib.motion_engine import MotionEngine
        engine = MotionEngine(
            project_name=self.project_name,
            projects_root=str(self.project_dir.parent),
            fps=24,
            resolution=self.resolution,
        )
        engine.render_episode(self.episode)

        self.analytics.end_stage("motion")

        self.state.mark("motion_complete", True)
        self._log("Stage 3b complete ✅")

    # ── Stage 4: Video Composition (FFmpeg, CPU only) ─────────────────────────

    def _stage4_compose(self) -> Optional[Path]:
        if self.state.get("video_complete"):
            self._log("Stage 4 already complete — skipping")
            output = self.project_dir / "output" / f"episode_{self.episode}.mp4"
            return output if output.exists() else None

        self._log("=== STAGE 4: Video Composition (FFmpeg) ===")

        self.analytics.start_stage("compose")
        from lib.video_composer import VideoComposer
        composer = VideoComposer(
            project_name=self.project_name,
            projects_root=str(self.project_dir.parent),
            resolution=self.resolution,
        )
        output_path = composer.compose_episode(self.episode)
        self.analytics.end_stage("compose")

        self.state.mark("video_complete", True)
        self._log(f"Stage 4 complete ✅ — {output_path}")
        return output_path

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """
        Run the full pipeline with staged model loading.
        Resumes from last completed stage automatically.
        """
        self._log(f"Starting pipeline (mode={self.mode}, episode={self.episode})")
        self._log(f"Current state:\n{self.state}")

        if self.input_file and Path(self.input_file).is_file() and Path(self.input_file).suffix.lower() in [".zip", ".cbz"]:
            import zipfile
            extract_dir = self.project_dir / "cache" / "extracted_input"
            extract_dir.mkdir(parents=True, exist_ok=True)
            self._log(f"Extracting {self.input_file} to {extract_dir} ...")
            with zipfile.ZipFile(self.input_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            self.input_file = str(extract_dir)


        if self.mode == "manhwa":
            self._stage0_manhwa_ocr()
            self._stage1_storyboard()
            self._stage2_images()
            self._stage3_audio()
            self._stage3b_motion()
            
        elif self.mode == "manhwa_panels":
            self._stage0_manhwa_panels()
            # Skip Stage 1 (Storyboard), Stage 2 (Images), Stage 3 (Audio)
            # The synthetic storyboard from Stage 0 is enough for Motion Engine
            self._stage3b_motion()
            
        else:
            # Normal novel mode
            self._stage1_storyboard()
            self._stage2_images()
            self._stage3_audio()
            self._stage3b_motion()
            
        output = self._stage4_compose()

        # Build Manifest & Analytics
        from lib.manifest_builder import ManifestBuilder
        from lib.metadata_generator import MetadataGenerator
        from lib.thumbnail_generator import ThumbnailGenerator
        from lib.review_generator import ReviewGenerator
        from lib.quality_checker import QualityChecker
        
        qc = QualityChecker(str(self.project_dir.parent))
        report = qc.validate_episode(self.project_name, self.episode)
        
        self.analytics.save()

        mb = ManifestBuilder(self.project_name, str(self.project_dir.parent))
        mb.build(
            episode=self.episode,
            pipeline_state=self.state._state,
            quality_report=report,
            analytics=self.analytics.compute()
        )
        
        mg = MetadataGenerator(self.project_name, str(self.project_dir.parent))
        mg.generate(episode=self.episode)
        
        tg = ThumbnailGenerator(self.project_name, str(self.project_dir.parent))
        tg.generate(episode=self.episode, title=f"{self.project_name} - Episode {self.episode}")
        
        rg = ReviewGenerator(self.project_name, str(self.project_dir.parent))
        rg.generate(episode=self.episode, quality_report=report)

        return {
            "project": self.project_name,
            "episode": self.episode,
            "output": str(output) if output else None,
            "state": self.state._state,
        }

#!/usr/bin/env python3
"""
run_pipeline.py — Main CLI entry point for CC-Novel2Video free pipeline.

Usage examples:
  # 1. Full novel pipeline
  python run_pipeline.py --input my_novel.txt --project fantasy_01 --mode novel

  # 2. Manhwa OCR pipeline
  python run_pipeline.py --input chapters/ch01/ --project webtoon_01 --mode manhwa

  # 3. Manhwa panel animation (skip image generation)
  python run_pipeline.py --input chapters/ch01/ --project webtoon_02 --mode manhwa_panels

  # 4. Resume an interrupted project
  python run_pipeline.py --project fantasy_01 --resume

  # 5. Run only specific stages
  python run_pipeline.py --project fantasy_01 --stage audio
  python run_pipeline.py --project fantasy_01 --stage compose --bgm music.mp3
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from lib.config import DEFAULT_LLM

# Configure logging before importing local modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cli")


def _setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CC-Novel2Video — Free Kaggle-Compatible Long-Form Video Factory"
    )

    # Core required
    parser.add_argument(
        "--project", type=str, required=True,
        help="Project name (creates folder in projects/)"
    )

    # Inputs and Modes
    parser.add_argument(
        "--input", type=str,
        help="Path to source text file (novel) or directory of images (manhwa)"
    )
    parser.add_argument(
        "--mode", type=str, choices=["novel", "manhwa", "manhwa_panels"], default="novel",
        help="Pipeline mode"
    )
    parser.add_argument(
        "--style", type=str, default="default",
        help="Visual style profile to use (matches a filename in styles/ directory, e.g. 'manhwa')"
    )
    parser.add_argument(
        "--episode", type=int, default=1,
        help="Episode number to generate (default: 1)"
    )

    # Stage control
    parser.add_argument(
        "--stage", type=str, choices=["all", "storyboard", "images", "audio", "compose"], default="all",
        help="Run only a specific pipeline stage"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an interrupted pipeline from project_state.json"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing files (disable resume for current stage)"
    )

    # Model & generation settings
    parser.add_argument(
        "--llm", type=str, default=DEFAULT_LLM,
        help=f"LLM backend: qwen2.5-7b, deepseek-r1-7b (default: {DEFAULT_LLM})"
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization for LLM (requires 24GB+ VRAM)"
    )
    parser.add_argument(
        "--resolution", type=str, choices=["1080p", "1440p", "4k", "16:9"], default="1080p",
        help="Output video resolution"
    )
    parser.add_argument(
        "--voice", type=str, default=os.environ.get("CCNV_VOICE", "af_heart"),
        help="Kokoro TTS voice (e.g., af_heart) OR edge-tts voice (e.g., en-US-AndrewNeural)"
    )

    # Output adjustments
    parser.add_argument(
        "--no-subtitles", action="store_true",
        help="Disable subtitle burning during compose stage"
    )
    parser.add_argument(
        "--bgm", type=str,
        help="Path to background music file to mix into the final video"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip actual model inference (for testing pipeline flow)"
    )
    parser.add_argument(
        "--fast-cpu-overlap", action="store_true",
        help="Use BatchScheduler to overlap CPU tasks (TTS/Motion) with GPU generation"
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None, metavar="N",
        help="Process only the first N storyboard scenes (debug mode — e.g. --max-scenes 5)"
    )

    return parser.parse_args()


def main():
    args = _setup_args()

    from lib.agents.video_agent import PipelineState
    from lib.batch_scheduler import BatchScheduler
    from lib.project_manager import ProjectManager
    from lib.services.pipeline_service import (
        run_pipeline,
        run_storyboard_stage,
        run_image_stage,
        run_audio_stage,
        run_compose_stage
    )

    pm = ProjectManager(Path("projects"))

    # If an input file is provided, copy it to the project source directory
    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            logger.error(f"Input path not found: {in_path}")
            sys.exit(1)

        # Initialize project if it doesn't exist
        try:
            pm.load_project(args.project)
        except Exception:
            logger.info(f"Creating new project: {args.project}")
            pm.create_project(args.project)

        project_dir = pm.get_project_path(args.project)
        source_dir = project_dir / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        if args.mode == "novel":
            if in_path.is_file():
                dest = source_dir / f"episode_{args.episode}.txt"
                import shutil
                shutil.copy2(in_path, dest)
                logger.info(f"Copied input to {dest}")
                # We update the argument so run_pipeline uses the local copy
                args.input = str(dest)
            else:
                logger.error("Novel mode requires a text file as --input")
                sys.exit(1)
        elif args.mode in ["manhwa", "manhwa_panels"]:
            if in_path.is_dir():
                logger.info(f"Using chapter directory: {in_path}")
            else:
                logger.error("Manhwa mode requires a directory of images as --input")
                sys.exit(1)

    elif args.stage == "all" and not args.resume:
        # If no input is given and we're not resuming, we can't start a full pipeline
        # (unless the file is already in source/episode_N.txt, which we could check,
        # but requiring --input is safer for UX).
        logger.warning(
            "No --input provided and --resume not set. "
            "Pipeline will attempt to find existing source files."
        )

    load_in_4bit = not args.no_4bit

    try:
        # ── Run Specific Stage ────────────────────────────────────────────────
        if args.stage == "storyboard":
            logger.info("=== Running Stage 1: Storyboard Only ===")
            run_storyboard_stage(
                project_name=args.project,
                episode=args.episode,
                llm=args.llm,
                overwrite=args.overwrite,
            )

        elif args.stage == "images":
            logger.info("=== Running Stage 2: Images Only ===")
            run_image_stage(
                project_name=args.project,
                episode=args.episode,
                resolution=args.resolution,
                dry_run=args.dry_run,
                max_scenes=args.max_scenes,
            )

        elif args.stage == "audio":
            logger.info("=== Running Stage 3: Audio Only ===")
            run_audio_stage(
                project_name=args.project,
                episode=args.episode,
                voice=args.voice,
            )

        elif args.stage == "compose":
            logger.info("=== Running Stage 4: Compose Only ===")
            run_compose_stage(
                project_name=args.project,
                episode=args.episode,
                resolution=args.resolution,
                include_subtitles=not args.no_subtitles,
                bgm_path=args.bgm,
            )

        # ── Run Full Pipeline ─────────────────────────────────────────────────
        else:
            logger.info(f"=== Starting Full Pipeline: mode={args.mode} ===")
            result = run_pipeline(
                project_name=args.project,
                input_file=args.input,
                mode=args.mode,
                episode=args.episode,
                projects_root=None,
                llm=args.llm,
                load_in_4bit=load_in_4bit,
                resolution=args.resolution,
                style=args.style,
                dry_run=args.dry_run,
                resume=args.resume or not args.overwrite,
                fast_cpu_overlap=args.fast_cpu_overlap,
                max_scenes=args.max_scenes,
            )
            
            if result.get("output_path"):
                logger.info(f"\n🎉 Pipeline complete! Video saved to: {result['output_path']}")
            else:
                logger.warning("\nPipeline finished, but output_path was not returned.")

    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user. State is saved; use --resume to continue.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

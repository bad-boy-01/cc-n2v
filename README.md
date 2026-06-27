# CC-Novel2Video (Kaggle-Compatible Free Pipeline)

A fully free, open-source, Kaggle-compatible long-form video factory that turns novels or manhwa/mangas into recap videos using local AI models.

## What is it?
This project automates the creation of 30-minute to 2-hour AI-generated story videos, novel videos, audiobook videos, and manhwa recap videos entirely with free, local models.

It has been completely refactored to run perfectly inside Kaggle's free 2x T4 GPU environment. **No paid APIs, no OpenAI, no Gemini, no Claude, no Veo required.**

## Pipeline Architecture
The system uses strict VRAM management (staged model loading) to ensure no Out-Of-Memory errors on 15GB GPUs. Only one model is active in VRAM at a time:

- **Intelligence Layer**: Uses a Director-led approach. The `PlannerAgent` chunks the story, the `DirectorAgent` specifies pacing/style, and the `StoryboardAgent` enforces validation using a multi-step `PromptOptimizer` and `PromptValidator`. Maintains continuity via the `WorldEngine` character state tracker.
- **LLM Layer (Stage 1)**: Qwen2.5-7B (or DeepSeek-R1-Distill) via bitsandbytes 4-bit quantization for storyboard extraction and character design.
- **Image Generation (Stage 2)**: FLUX.1-schnell (or SDXL fallback) loaded via a decoupled Plugin architecture.
- **Voice / TTS (Stage 3)**: Kokoro TTS for high-quality narrative and dialogue voices.
- **Motion & Video Composition (Stage 4)**: A custom CPU-based Ken Burns + pan effect engine combined with FFmpeg automation. Supports `--fast-cpu-overlap` via `BatchScheduler` to execute this asynchronously during GPU rendering.
- **Quality Layer**: Automatically generates `QualityReports` simulating VQA, compiles `PipelineAnalytics`, and builds detailed metadata manifests.

## Key Features
- 🎬 **Multiple Generation Modes**:
  - `novel`: Standard text-to-video workflow.
  - `manhwa`: Extracts text from manhwa pages via OCR (EasyOCR/PaddleOCR) and regenerates scenes using FLUX.
  - `manhwa_panels`: Extracts panels directly via OpenCV and animates the original artwork (Fastest & perfectly preserves original art).
- 💾 **Crash Resumption**: Deep checkpointing saves state after every stage (`project_state.json`), enabling safe resumption of massive 2-hour long video renders.
- 🎨 **Character Consistency**: Uses reference image caching and targeted prompt injection rather than heavy ControlNets.
- 🚀 **Kaggle Ready**: Includes a bootstrap script to get up and running instantly on Kaggle notebooks.
- 🖥️ **WebUI**: Includes a FastAPI WebUI for visualizing generated assets, monitoring pipeline progress, and modifying scripts.

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/bad-boy-01/cc-n2v.git
cd cc-n2v
```

2. **Kaggle Setup / Quickstart (Recommended):**
Open a new notebook in Kaggle (T4 x2 environment), open the console or add code cells, and run the following bootstrap commands:

```bash
# 1. Clone the repository and install dependencies
!git clone https://github.com/bad-boy-01/cc-n2v.git
%cd cc-n2v
!bash setup_kaggle.sh

# 2. Setup Ollama locally (Highly Recommended to save disk space & VRAM)
!curl -fsSL https://ollama.com/install.sh | sh
!nohup ollama serve > ollama.log 2>&1 &
!sleep 3
!ollama pull qwen2.5:7b

# 3. Run the pipeline!
!USE_OLLAMA=1 python run_pipeline.py --project my_story --input projects/my_story/source/chapter1.txt --mode novel
```

*(For local Windows/Linux setup, simply run `pip install -r requirements.txt` and ensure FFmpeg is installed on your system).*

## Usage

Use the CLI orchestrator to run the pipeline. 

**Standard Novel-to-Video Mode (with CPU Overlap):**
```bash
python run_pipeline.py --project my_story --input source_novel.txt --mode novel --fast-cpu-overlap
```

**Manhwa Panel Animation Mode (Recap Channels):**
```bash
python run_pipeline.py --project recap_01 --input chapters/ch01/ --mode manhwa_panels
```

**Resume an interrupted generation:**
```bash
python run_pipeline.py --project my_story --resume
```

**Run a specific stage only:**
```bash
python run_pipeline.py --project my_story --stage audio
```

## Project Directory Structure

```
projects/my_story/
├── source/       # Source text or chapter images
├── scripts/      # Generated storyboard JSON
├── characters/   # Character reference designs
├── clues/        # Important prop/location designs
├── images/       # Generated scenes or cropped panels
├── audio/        # Synthesized TTS wav files + durations
├── cache/        # Intermediate motion MP4 clips
└── output/       # Final rendered video (.mp4)
```

## License
[AGPL-3.0](LICENSE)

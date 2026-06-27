#!/bin/bash
# setup_kaggle.sh — Bootstrap script for Kaggle environments (2x T4 GPUs)
#
# Usage inside a Kaggle notebook cell:
#   !bash setup_kaggle.sh

set -e

echo "=========================================================="
echo "🚀 Setting up CC-Novel2Video Free Kaggle Pipeline 🚀"
echo "=========================================================="

# 1. Update package list and install system dependencies
echo "=> Installing system dependencies (FFmpeg, espeak)..."
sudo apt-get update -y
sudo apt-get install -y ffmpeg espeak-ng

# 2. Install Python requirements
echo "=> Installing Python dependencies..."
pip install -r requirements.txt

# 3. Create required directory structure
echo "=> Creating project directories..."
mkdir -p projects/ cache/ output/

# 4. Set Kaggle-specific environment variables in a .env file (optional, but good practice)
echo "=> Configuring environment variables for Kaggle (T4 15GB VRAM limits)..."
cat << EOF > .env
CCNV_KAGGLE=true
CCNV_LLM=qwen2.5-7b
CCNV_LLM_4BIT=true
CCNV_IMAGE_MODEL=flux_schnell
CCNV_BATCH_SIZE=25
CCNV_IP_ADAPTER=false
EOF
echo ".env file created."

# 5. Pre-download models (Optional but recommended to prevent timeouts during execution)
# Note: Kokoro, Qwen2.5, and FLUX will download on first run automatically via huggingface_hub.
# If you want to force download them here, you can add Python scripts to load them, 
# but lazy-loading handles it fine.

echo "=========================================================="
echo "✅ Setup Complete!"
echo ""
echo "To run the pipeline:"
echo "  python run_pipeline.py --project my_novel --input my_novel.txt"
echo "=========================================================="

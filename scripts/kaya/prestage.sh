#!/bin/bash --login
# One-time, on the LOGIN node (internet): pre-download the dataset + all model
# weights into the /group HF cache so compute nodes (offline) can load them.
# See kaya_cheatsheet.md §"Downloading models & data".
#
#   bash scripts/kaya/prestage.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$HERE/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$MPVRDU_ENV"

mkdir -p "$HF_HOME" "$(dirname "$MPVRDU_MMLB_DIR")"

echo "== dataset -> $MPVRDU_MMLB_DIR =="
python "$REPO/scripts/download_data.py" --out "$MPVRDU_MMLB_DIR"

echo "== model weights -> $HF_HOME =="
# Generators (pick what you'll run; 32B optional/large).
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
# huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct
# Dense text encoder.
huggingface-cli download sentence-transformers/all-mpnet-base-v2
# Visual retrievers (adapters + their base models, both needed offline).
huggingface-cli download vidore/colpali-v1.3
huggingface-cli download vidore/colpaligemma-3b-pt-448-base
huggingface-cli download vidore/colqwen2.5-v0.2
huggingface-cli download vidore/colqwen2.5-base

echo "prestage complete. HF_HOME=$HF_HOME"
du -sh "$HF_HOME" "$MPVRDU_MMLB_DIR"

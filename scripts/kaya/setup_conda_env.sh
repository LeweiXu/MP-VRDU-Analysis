#!/bin/bash --login
# One-time, on the LOGIN node (it has internet): create the conda env under
# /group and install dependencies. See kaya_cheatsheet.md §"Conda environment".
#
#   bash scripts/kaya/setup_conda_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

module load "$MPVRDU_ANACONDA"
module load "$MPVRDU_CUDA"

mkdir -p "$(dirname "$MPVRDU_ENV")"
conda create -p "$MPVRDU_ENV" python=3.11 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$MPVRDU_ENV"

# Install torch matching the cluster CUDA module FIRST (edit the index-url to the
# cu<version> that matches $MPVRDU_CUDA — get the exact line from pytorch.org).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then the project deps.
REPO="$(cd "$HERE/../.." && pwd)"
pip install -r "$REPO/requirements-gpu.txt"

echo "env ready at $MPVRDU_ENV"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"

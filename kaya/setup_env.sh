#!/bin/bash --login
# One-time, on the LOGIN node (it has internet): create a conda env under
# /group with torch + transformers + the Qwen-VL helper package.
#
#   bash setup_env.sh  (run on Kaya, from $KAYA_REMOTE_DIR)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

load_modules

mkdir -p "$(dirname "$KAYA_ENV")"
if [[ ! -d "$KAYA_ENV/conda-meta" ]]; then
  conda create -p "$KAYA_ENV" python=3.11 -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$KAYA_ENV"
python -m pip install --upgrade pip wheel setuptools

# Match torch to the cluster's CUDA module (set KAYA_TORCH_INDEX_URL above).
python -m pip install "torch>=2.3" --index-url "$KAYA_TORCH_INDEX_URL"

# Transformers + helpers for both text-only and Qwen-VL models.
python -m pip install transformers accelerate pillow qwen-vl-utils
python -m pip check

echo "env ready at $KAYA_ENV"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| CUDA build', torch.version.cuda)"

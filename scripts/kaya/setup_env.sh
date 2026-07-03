#!/bin/bash --login
# Build the MP-VRDU conda environment on the Kaya login node.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

load_modules

mkdir -p "$(dirname "$KAYA_ENV")" "$HF_HOME"
if [[ ! -d "$KAYA_ENV/conda-meta" ]]; then
  conda create -p "$KAYA_ENV" python=3.11 -y
fi

activate_env
python -m pip install --upgrade pip wheel setuptools
python -m pip install --extra-index-url "$KAYA_TORCH_INDEX_URL" -r "$KAYA_REMOTE_DIR/requirements.txt"
python -m pip check

echo "Environment ready at $KAYA_ENV"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

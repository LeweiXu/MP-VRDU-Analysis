#!/bin/bash --login
# Download v1 models and MMLongBench-Doc into the Kaya mirror cache.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

load_modules
activate_env

mkdir -p "$HF_HOME" "$KAYA_DATA_DIR"

MODEL_IDS_DEFAULT=(
  "Qwen/Qwen3-VL-2B-Instruct"
  "Qwen/Qwen3-VL-4B-Instruct"
  "Qwen/Qwen3-VL-8B-Instruct"
  "Qwen/Qwen3-VL-32B-Instruct"
)

if [[ -n "${MPVRDU_MODEL_IDS:-}" ]]; then
  read -r -a MODEL_IDS <<< "$MPVRDU_MODEL_IDS"
else
  MODEL_IDS=("${MODEL_IDS_DEFAULT[@]}")
fi

for model_id in "${MODEL_IDS[@]}"; do
  python "$KAYA_REMOTE_DIR/scripts/kaya/download_hf.py" --model "$model_id"
done

python "$KAYA_REMOTE_DIR/scripts/kaya/download_hf.py" --dataset "yubo2333/MMLongBench-Doc"

echo "Prestage complete."
du -sh "$HF_HOME" 2>/dev/null || true

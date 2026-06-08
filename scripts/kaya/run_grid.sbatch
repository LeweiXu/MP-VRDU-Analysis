#!/bin/bash --login
# Run the full experiment grid on Kaya with the real Qwen2.5-VL (7B by default).
# Reuses the SAME suite as local; the generator is overridden on the CLI, so the
# only difference from local is this one flag. Resumable — resubmit to continue.
#   sbatch scripts/kaya/run_grid.sbatch
#SBATCH --job-name=mpvrdu_grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$HERE/env.sh"
load_modules
activate_env
set_offline                      # compute node = no internet

MODEL="${MPVRDU_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"   # or -32B
echo "grid with generator=kaya_vlm model=$MODEL  HF_HOME=$HF_HOME"

cd "$REPO"
python scripts/run_grid.py \
  --suite experiments/grid_local_3b.yaml \
  --generator kaya_vlm --model-id "$MODEL" \
  --results-root "$MPVRDU_RESULTS/grid"

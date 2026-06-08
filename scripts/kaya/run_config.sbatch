#!/bin/bash --login
# Run a single config on Kaya.  sbatch scripts/kaya/run_config.sbatch <config.yaml>
# The full-benchmark configs (configs/oracle.yaml, configs/subA/*, ...) already
# specify generator: kaya_vlm, so no override is needed.
#SBATCH --job-name=mpvrdu_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
CONFIG="${1:?usage: sbatch run_config.sbatch <config.yaml>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$HERE/env.sh"
load_modules
activate_env
set_offline

cd "$REPO"
python -m mpvrdu.pipeline --config "$CONFIG" \
  --out "$MPVRDU_RESULTS/$(basename "${CONFIG%.yaml}").jsonl"

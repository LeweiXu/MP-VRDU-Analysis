#!/bin/bash --login
# Run a Qwen inference smoke test on a GPU node.
# Usually invoked via `bash sync_kaya.sh run run_qwen.sbatch [args...]` from kaya/.
#   sbatch run_qwen.sbatch
#   sbatch run_qwen.sbatch --model Qwen/Qwen2.5-VL-3B-Instruct \
#     --prompt "Describe this image." --image /path/to/image.png
#SBATCH --job-name=qwen_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
HERE="$SLURM_SUBMIT_DIR"
source "$HERE/env.sh"
load_modules
activate_env
set_offline

python "$HERE/qwen_infer.py" "$@"

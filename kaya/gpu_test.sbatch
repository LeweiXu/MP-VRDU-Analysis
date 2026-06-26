#!/bin/bash --login
# Smoke test: is the GPU stack alive? Submit FIRST, before any model run.
#   sbatch gpu_test.sbatch  (from kaya/, after `sync_kaya.sh push`)
#SBATCH --job-name=qwen_gpu_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

HERE="$SLURM_SUBMIT_DIR"
source "$HERE/env.sh"
load_modules
activate_env

python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

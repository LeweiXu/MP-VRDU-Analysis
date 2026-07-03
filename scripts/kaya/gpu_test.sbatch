#!/bin/bash --login
#SBATCH --job-name=mpvrdu_gpu_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

HERE="$SLURM_SUBMIT_DIR"
source "$HERE/scripts/kaya/env.sh"

load_modules
activate_env

python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

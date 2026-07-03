#!/bin/bash --login
#SBATCH --job-name=mpvrdu_experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

HERE="$SLURM_SUBMIT_DIR"
source "$HERE/scripts/kaya/env.sh"

load_modules
activate_env
set_offline

cd "$KAYA_REMOTE_DIR"
python -m cli.run_experiment "$@"

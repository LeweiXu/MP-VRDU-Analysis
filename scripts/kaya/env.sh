#!/bin/bash
# Shared Kaya environment for the MP-VRDU pipeline.
#
# Source this file from local sync commands and from Kaya login/compute jobs.

_mpvrdu_root_from_script() {
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd
}

export MPVRDU_ROOT="${MPVRDU_ROOT:-$(_mpvrdu_root_from_script)}"

# Account and module defaults. Override these in your shell if Kaya changes.
export KAYA_HOST="${KAYA_HOST:-kaya.hpc.uwa.edu.au}"
export KAYA_USER="${KAYA_USER:-lxu}"
export KAYA_PROJECT="${KAYA_PROJECT:-ems036}"
export KAYA_SSH_ALIAS="${KAYA_SSH_ALIAS:-kaya}"
export KAYA_ANACONDA="${KAYA_ANACONDA:-Anaconda3/2024.06}"
export KAYA_CUDA="${KAYA_CUDA:-cuda/12.6.3}"
export KAYA_TORCH_INDEX_URL="${KAYA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
export KAYA_GPU_PARTITION="${KAYA_GPU_PARTITION:-gpu}"
export KAYA_GPU_GRES="${KAYA_GPU_GRES:-gpu:1}"

export MYGROUP="${MYGROUP:-/group/$KAYA_PROJECT/$KAYA_USER}"
export KAYA_REMOTE_DIR="${KAYA_REMOTE_DIR:-$MYGROUP/mpvrdu}"

# Root-relative artifact paths on the Kaya mirror.
export HF_HOME="${HF_HOME:-$KAYA_REMOTE_DIR/.cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TORCH_HOME="${TORCH_HOME:-$HF_HOME/torch}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HF_HOME/pip}"
export KAYA_ENV="${KAYA_ENV:-$KAYA_REMOTE_DIR/envs/mpvrdu}"
export KAYA_DATA_DIR="${KAYA_DATA_DIR:-$KAYA_REMOTE_DIR/.data}"
export KAYA_RESULTS_DIR="${KAYA_RESULTS_DIR:-$KAYA_REMOTE_DIR/results}"
export KAYA_LOGS_DIR="${KAYA_LOGS_DIR:-$KAYA_REMOTE_DIR/logs}"

load_modules() {
  module load gcc/9.4.0 2>/dev/null || true
  module load "$KAYA_ANACONDA"
  module load "$KAYA_CUDA"
  module list 2>&1 | sed 's/^/[modules] /'
}

activate_env() {
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda deactivate 2>/dev/null || true
  conda activate "$KAYA_ENV"
}

set_offline() {
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

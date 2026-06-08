#!/bin/bash
# Shared Kaya environment — sourced by the setup, prestage and sbatch scripts.
# Set these once for your project (here, in ~/.bashrc, or export before calling).
#
# Everything lives under /group (has space); $HOME is small (kaya_cheatsheet.md).

# --- EDIT THESE for your project --------------------------------------------
export MPVRDU_GROUP="${MPVRDU_GROUP:-/group/CHANGE_ME}"   # your project dir
export MPVRDU_CUDA="${MPVRDU_CUDA:-cuda/12.1}"            # match the torch build
export MPVRDU_ANACONDA="${MPVRDU_ANACONDA:-Anaconda3/2021.05}"
# ---------------------------------------------------------------------------

# Derived paths (all under /group)
export MPVRDU_ENV="${MPVRDU_ENV:-$MPVRDU_GROUP/conda_environments/mpvrdu}"
export MPVRDU_CACHE="${MPVRDU_CACHE:-$MPVRDU_GROUP/mpvrdu_cache}"      # torch hub etc.
export HF_HOME="${HF_HOME:-$MPVRDU_GROUP/hf_cache}"                   # HF models+datasets
export MPVRDU_MMLB_DIR="${MPVRDU_MMLB_DIR:-$MPVRDU_GROUP/data/mmlongbench}"
export MPVRDU_RENDER_CACHE="${MPVRDU_RENDER_CACHE:-$MPVRDU_GROUP/mpvrdu_cache/renders}"
export MPVRDU_RESULTS="${MPVRDU_RESULTS:-$MPVRDU_GROUP/mpvrdu_results}"

if [[ "$MPVRDU_GROUP" == *CHANGE_ME* ]]; then
  echo "!! Set MPVRDU_GROUP to your /group project path (see scripts/kaya/env.sh)" >&2
fi

load_modules() {
  module load gcc/9.4.0 2>/dev/null || true
  module load "$MPVRDU_ANACONDA"
  module load "$MPVRDU_CUDA"
  module list 2>&1 | sed 's/^/[modules] /'
}

activate_env() {
  # conda activate by absolute path (env lives under /group)
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate "$MPVRDU_ENV"
}

# On compute nodes there is NO internet — force offline so nothing phones home.
set_offline() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
}

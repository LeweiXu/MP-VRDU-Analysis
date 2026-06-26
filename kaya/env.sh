#!/bin/bash
# Generic Kaya environment for the standalone setup/testing scripts in this
# folder. Independent of the main mpvrdu pipeline.
#
# `source` this file before doing anything else (on Kaya, or locally for
# sync_kaya.sh).

# --- EDIT THESE for your account ---------------------------------------------
export KAYA_HOST="${KAYA_HOST:-kaya.hpc.uwa.edu.au}"
export KAYA_USER="${KAYA_USER:-lxu}"
export KAYA_PROJECT="${KAYA_PROJECT:-ems036}"
export KAYA_CUDA="${KAYA_CUDA:-cuda/12.6.3}"
export KAYA_ANACONDA="${KAYA_ANACONDA:-Anaconda3/2024.06}"
export KAYA_TORCH_INDEX_URL="${KAYA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
# -------------------------------------------------------------------------

# Matches Kaya's own ~/.kaya_env.sh ($MYGROUP)
export MYGROUP="${MYGROUP:-/group/$KAYA_PROJECT/$KAYA_USER}"
# ssh alias for $KAYA_USER@$KAYA_HOST — set up `Host kaya` in ~/.ssh/config
# (see kaya/README.md §0b) so this is passwordless.
export KAYA_SSH_ALIAS="${KAYA_SSH_ALIAS:-kaya}"
# Where the contents of this kaya/ directory are synced to on Kaya (see sync_kaya.sh)
export KAYA_REMOTE_DIR="${KAYA_REMOTE_DIR:-$MYGROUP/kaya_test}"

export KAYA_ENV="${KAYA_ENV:-$MYGROUP/conda_environments/qwen_demo}"
export HF_HOME="${HF_HOME:-$MYGROUP/hf_cache}"

load_modules() {
  module load gcc/9.4.0 2>/dev/null || true
  module load "$KAYA_ANACONDA"
  module load "$KAYA_CUDA"
  module list 2>&1 | sed 's/^/[modules] /'
}

activate_env() {
  source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
  # `conda activate` is a no-op if $KAYA_ENV is already the active env, but
  # `module load` (in load_modules) re-prepends its own bin/ to PATH every
  # time regardless — so a stale activation can end up shadowed by the
  # module's python. Deactivate first to force a real (re-)activation that
  # puts $KAYA_ENV/bin back at the front of PATH.
  conda deactivate 2>/dev/null || true
  conda activate "$KAYA_ENV"
}

# Compute nodes have NO internet — force offline so HF doesn't try to phone home.
set_offline() {
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

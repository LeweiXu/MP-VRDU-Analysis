#!/bin/bash
# Print every variable env.sh resolves to, and sanity-check the environment.
# Safe to run both locally and on Kaya (login or compute node) — it adapts
# its checks based on where it's running.
#
#   bash check_env.sh           # locally
#   ssh kaya bash -lc "'cd \$KAYA_REMOTE_DIR && bash check_env.sh'"
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

echo "== resolved variables (from env.sh) =="
for v in KAYA_HOST KAYA_USER KAYA_PROJECT KAYA_CUDA KAYA_ANACONDA \
         KAYA_TORCH_INDEX_URL MYGROUP KAYA_SSH_ALIAS KAYA_REMOTE_DIR \
         KAYA_ENV HF_HOME; do
  printf '%-22s = %s\n' "$v" "${!v}"
done

echo
echo "== host check =="
host="$(hostname)"
echo "hostname: $host"

if [[ "$host" == kaya* || "$host" == k0* || "$host" == k1* ]]; then
  echo "Running ON KAYA."

  if [[ -d "$MYGROUP" ]]; then
    echo "OK   MYGROUP exists: $MYGROUP"
  else
    echo "MISS MYGROUP does not exist: $MYGROUP (check KAYA_PROJECT/KAYA_USER, or 'quota -s')"
  fi

  if [[ -d "$KAYA_REMOTE_DIR" ]]; then
    echo "OK   KAYA_REMOTE_DIR exists: $KAYA_REMOTE_DIR"
  else
    echo "MISS KAYA_REMOTE_DIR does not exist yet: $KAYA_REMOTE_DIR"
    echo "     -> run 'bash sync_kaya.sh push' from your LOCAL machine first"
  fi

  if [[ -d "$KAYA_ENV/conda-meta" ]]; then
    echo "OK   conda env exists: $KAYA_ENV"
  else
    echo "MISS conda env not created yet: $KAYA_ENV"
    echo "     -> run 'bash setup_env.sh' (on the login node)"
  fi

  if [[ -d "$HF_HOME" ]]; then
    size="$(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
    echo "OK   HF_HOME exists: $HF_HOME ($size)"
  else
    echo "MISS HF_HOME does not exist yet: $HF_HOME"
    echo "     -> run 'bash prestage.sh' or 'python download_hf.py <repo>'"
  fi

  if curl -sI --max-time 3 https://huggingface.co >/dev/null 2>&1; then
    echo "INFO internet: reachable (you're on the LOGIN node — OK to download)"
  else
    echo "INFO internet: NOT reachable (you're on a COMPUTE node — expected; use set_offline)"
  fi

else
  echo "Running LOCALLY."

  if command -v rsync >/dev/null; then
    echo "OK   rsync found"
  else
    echo "MISS rsync not found — install it (apt/brew install rsync)"
  fi

  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$KAYA_SSH_ALIAS" true 2>/dev/null; then
    echo "OK   ssh $KAYA_SSH_ALIAS works without a password"
  else
    echo "MISS ssh $KAYA_SSH_ALIAS failed or needs a password"
    echo "     -> check ~/.ssh/config has a 'Host $KAYA_SSH_ALIAS' entry (see README §0b)"
  fi
fi

#!/bin/bash --login
# One-time, on the LOGIN node (internet): download small Qwen models into the
# /group HF cache so compute nodes (offline) can load them.
#
#   bash prestage.sh  (run on Kaya, from $KAYA_REMOTE_DIR)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
load_modules
activate_env

mkdir -p "$HF_HOME"

hf_download() {
  python -c \
    "from huggingface_hub import snapshot_download; import sys; print(snapshot_download(sys.argv[1]))" \
    "$1"
}

echo "== text model -> $HF_HOME =="
hf_download Qwen/Qwen2.5-1.5B-Instruct

echo "== vision-language model -> $HF_HOME =="
hf_download Qwen/Qwen2.5-VL-3B-Instruct

echo "prestage complete. HF_HOME=$HF_HOME"
du -sh "$HF_HOME"

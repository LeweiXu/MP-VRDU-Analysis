#!/bin/bash
# Sync a local directory to/from Kaya and submit/poll jobs over ssh (rsync),
# without going through GitHub or an interactive Kaya shell.
#
#   bash sync_kaya.sh push [local-dir]                # local-dir (default: cwd) -> $KAYA_REMOTE_DIR
#   bash sync_kaya.sh pull                            # logs/ and results/ <- Kaya
#   bash sync_kaya.sh pull-path <remote-relative-path> [local-dest]
#   bash sync_kaya.sh submit <sbatch-script> [args...]   # push, then sbatch
#   bash sync_kaya.sh watch [jobid]                   # block until job leaves the queue
#   bash sync_kaya.sh run <sbatch-script> [args...]   # submit + watch + pull
#
# Run from your local machine (not on Kaya). Requires `rsync` and a working
# `ssh kaya` (set up a key + ~/.ssh/config alias, see kaya/README.md §0b).
# <sbatch-script> paths are relative to the pushed directory (e.g. if you
# `push` from kaya/, just pass `gpu_test.sbatch`).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

REMOTE="$KAYA_SSH_ALIAS"
EXCLUDES=(--exclude .git --exclude .cache --exclude __pycache__ --exclude '*.pyc' --exclude logs --exclude results --exclude .kaya_last_job --exclude Kaya-ListerLab-Tutorial)
JOBFILE="$HERE/.kaya_last_job"

push() {
  local src="${1:-$(pwd)}"
  src="$(cd "$src" && pwd)"
  ssh "$REMOTE" "mkdir -p '$KAYA_REMOTE_DIR'"
  rsync -avz --delete "${EXCLUDES[@]}" "$src/" "$REMOTE:$KAYA_REMOTE_DIR/"
}

# Re-quote each argument so it survives the trip through `ssh "$REMOTE" "..."`
# as a single token (otherwise e.g. --prompt "two words" gets word-split
# remotely into --prompt two words).
quote_args() {
  local out=""
  for a in "$@"; do
    out+=" $(printf '%q' "$a")"
  done
  printf '%s' "$out"
}

pull() {
  for sub in logs results; do
    mkdir -p "$HERE/$sub"
    rsync -avz "$REMOTE:$KAYA_REMOTE_DIR/$sub/" "$HERE/$sub/" 2>/dev/null || true
  done
}

cmd="${1:-}"
case "$cmd" in
  push)
    push "${2:-}"
    ;;
  pull)
    pull
    ;;
  pull-path)
    remote_path="${2:?usage: sync_kaya.sh pull-path <remote-relative-path> [local-dest]}"
    local_dest="${3:-$HERE/${remote_path}}"
    mkdir -p "$(dirname "$local_dest")"
    rsync -avz "$REMOTE:$KAYA_REMOTE_DIR/$remote_path" "$local_dest"
    ;;
  submit)
    script="${2:?usage: sync_kaya.sh submit <sbatch-script> [sbatch-args...]}"
    shift 2
    push
    jobid=$(ssh "$REMOTE" "cd '$KAYA_REMOTE_DIR' && sbatch --parsable $script$(quote_args "$@")")
    echo "$jobid" > "$JOBFILE"
    echo "Submitted job $jobid"
    ;;
  watch)
    jobid="${2:-$(cat "$JOBFILE" 2>/dev/null || true)}"
    : "${jobid:?usage: sync_kaya.sh watch <jobid> (or run 'submit' first)}"
    echo "Waiting for job $jobid to leave the queue..."
    ssh "$REMOTE" "while squeue -h -j $jobid 2>/dev/null | grep -q .; do sleep 10; done"
    ssh "$REMOTE" "sacct -j $jobid --format=JobID,JobName,State,Elapsed,ExitCode --noheader"
    ;;
  run)
    script="${2:?usage: sync_kaya.sh run <sbatch-script> [sbatch-args...]}"
    shift 2
    push
    jobid=$(ssh "$REMOTE" "cd '$KAYA_REMOTE_DIR' && sbatch --parsable $script$(quote_args "$@")")
    echo "$jobid" > "$JOBFILE"
    echo "Submitted job $jobid, waiting..."
    ssh "$REMOTE" "while squeue -h -j $jobid 2>/dev/null | grep -q .; do sleep 10; done"
    ssh "$REMOTE" "sacct -j $jobid --format=JobID,JobName,State,Elapsed,ExitCode --noheader"
    pull
    echo "Logs synced to $HERE/logs/"
    ;;
  *)
    echo "usage: $0 {push [dir]|pull|pull-path <path> [dest]|submit <script> [args]|watch [jobid]|run <script> [args]}" >&2
    exit 1
    ;;
esac

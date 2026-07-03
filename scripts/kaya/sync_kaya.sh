#!/bin/bash
# Push the local repo to Kaya, submit jobs, wait for completion, and pull results.
#
#   bash scripts/kaya/sync_kaya.sh push
#   bash scripts/kaya/sync_kaya.sh pull
#   bash scripts/kaya/sync_kaya.sh submit scripts/kaya/gpu_test.sbatch
#   bash scripts/kaya/sync_kaya.sh run scripts/kaya/run_experiment.sbatch [args...]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/env.sh"

REMOTE="$KAYA_SSH_ALIAS"
JOBFILE="$ROOT/.kaya_last_job"

RSYNC_FILTERS=(
  --exclude ".git/"
  --exclude ".cache/"
  --exclude ".data/"
  --exclude "envs/"
  --exclude "results/"
  --exclude "logs/"
  --exclude "__pycache__/"
  --exclude "*.pyc"
)

push_repo() {
  local src="${1:-$ROOT}"
  src="$(cd "$src" && pwd)"
  ssh "$REMOTE" "mkdir -p '$KAYA_REMOTE_DIR' '$KAYA_RESULTS_DIR' '$KAYA_LOGS_DIR'"
  rsync -avz --delete "${RSYNC_FILTERS[@]}" "$src/" "$REMOTE:$KAYA_REMOTE_DIR/"
  ssh "$REMOTE" "mkdir -p '$KAYA_RESULTS_DIR' '$KAYA_LOGS_DIR'"
}

quote_args() {
  local out=""
  for arg in "$@"; do
    out+=" $(printf '%q' "$arg")"
  done
  printf '%s' "$out"
}

pull_results() {
  mkdir -p "$ROOT/results" "$ROOT/logs"
  rsync -avz "$REMOTE:$KAYA_RESULTS_DIR/" "$ROOT/results/" 2>/dev/null || true
  rsync -avz "$REMOTE:$KAYA_LOGS_DIR/" "$ROOT/logs/" 2>/dev/null || true
}

submit_job() {
  local script="${1:?usage: sync_kaya.sh submit <sbatch-script> [args...]}"
  shift
  push_repo
  local jobid
  jobid=$(ssh "$REMOTE" "cd '$KAYA_REMOTE_DIR' && sbatch --parsable $script$(quote_args "$@")")
  echo "$jobid" > "$JOBFILE"
  echo "Submitted job $jobid"
}

watch_job() {
  local jobid="${1:-$(cat "$JOBFILE" 2>/dev/null || true)}"
  : "${jobid:?usage: sync_kaya.sh watch <jobid> or submit first}"
  echo "Waiting for job $jobid to leave the queue..."
  ssh "$REMOTE" "while squeue -h -j $jobid 2>/dev/null | grep -q .; do sleep 10; done"
  ssh "$REMOTE" "sacct -j $jobid --format=JobID,JobName,State,Elapsed,ExitCode --noheader"
}

cmd="${1:-}"
case "$cmd" in
  push)
    push_repo "${2:-$ROOT}"
    ;;
  pull)
    pull_results
    ;;
  pull-path)
    remote_path="${2:?usage: sync_kaya.sh pull-path <remote-relative-path> [local-dest]}"
    local_dest="${3:-$ROOT/$remote_path}"
    mkdir -p "$(dirname "$local_dest")"
    rsync -avz "$REMOTE:$KAYA_REMOTE_DIR/$remote_path" "$local_dest"
    ;;
  submit)
    shift
    submit_job "$@"
    ;;
  watch)
    watch_job "${2:-}"
    ;;
  run)
    script="${2:?usage: sync_kaya.sh run <sbatch-script> [args...]}"
    shift 2
    submit_job "$script" "$@"
    watch_job
    pull_results
    echo "Results synced to $ROOT/results/ and logs synced to $ROOT/logs/"
    ;;
  *)
    echo "usage: $0 {push [dir]|pull|pull-path <path> [dest]|submit <sbatch-script> [args...]|watch [jobid]|run <sbatch-script> [args...]}" >&2
    exit 1
    ;;
esac

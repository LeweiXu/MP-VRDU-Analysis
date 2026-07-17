#!/usr/bin/env bash
# Detached: waits for the g2 resume (job 1065168) to end (or be killed at the
# 17:00 shutdown), pulls its final results, then judges + rebuilds g2. Normal
# resume is append-only so a shutdown kill loses nothing. Judging needs no Kaya.
set -uo pipefail
cd /home/lingwei/mpvrdu
PY=envs/mpvrdu/bin/python
LOG=logs/g2_final_watch.log
JOB=1065168
echo "================ g2 final watch start $(date) ================" >> "$LOG"
st=""
for i in $(seq 1 120); do
  st=$(ssh -o ConnectTimeout=20 kaya "sacct -j $JOB --format=State -X -n -P 2>/dev/null | head -1" 2>>"$LOG" | tr -d ' ')
  echo "$(date) job $JOB state=[$st]" >> "$LOG"
  case "$st" in RUNNING|PENDING|"") sleep 180 ;; *) break ;; esac
done
echo "$(date) job terminal ($st) -> pull" >> "$LOG"
"$PY" -m ops.kaya.kaya pull >> "$LOG" 2>&1 || echo "$(date) pull FAILED" >> "$LOG"
for i in $(seq 1 90); do pgrep -f "ops\.judge" >/dev/null 2>&1 || break; echo "$(date) waiting for a judge to finish" >> "$LOG"; sleep 60; done
"$PY" -m ops.judge --spec ops/specs/kaya_g2_full.yaml --judge-spec gemini-flash >> "$LOG" 2>&1 || echo "$(date) judge FAILED" >> "$LOG"
"$PY" -m ops.build --task G2_retrieval --run-tag g2-retrieval-full >> "$LOG" 2>&1 || echo "$(date) build FAILED" >> "$LOG"
"$PY" -m ops.mine >> "$LOG" 2>&1 || echo "$(date) mine FAILED" >> "$LOG"
"$PY" -m ops.scripts.build_status >> "$LOG" 2>&1 || echo "$(date) build_status FAILED" >> "$LOG"
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import json, collections
P = "results/cache/g2-retrieval-full/full/G2_retrieval/predictions.jsonl"
c = collections.Counter((json.loads(l).get("status") or "ok") for l in open(P) if l.strip())
print("g2 final:", dict(c), "total", sum(c.values()), "/15246")
PYEOF
echo "================ g2 final watch DONE $(date) ================" >> "$LOG"

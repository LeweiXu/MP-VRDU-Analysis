#!/usr/bin/env bash
# Detached: waits for the parser-recovery resume (job 1063881) to finish, pulls
# its results, then re-judges + rebuilds the unlimited parser arm so the 144
# recovered cells land in the tables. Survives session death. Judging needs no
# Kaya, so a VPN drop after the pull is harmless.
set -uo pipefail
cd /home/lingwei/mpvrdu
PY=envs/mpvrdu/bin/python
LOG=logs/parser_recover_watch.log
JOB=1063881
echo "================ parser-recover watch start $(date) ================" >> "$LOG"
st=""
for i in $(seq 1 120); do   # up to ~6h at 3-min spacing
  st=$(ssh -o ConnectTimeout=20 kaya "sacct -j $JOB --format=State -X -n -P 2>/dev/null | head -1" 2>>"$LOG" | tr -d ' ')
  echo "$(date) job $JOB state=[$st]" >> "$LOG"
  case "$st" in RUNNING|PENDING|"") sleep 180 ;; *) break ;; esac
done
echo "$(date) job terminal ($st) -> pull" >> "$LOG"
"$PY" -m ops.kaya.kaya pull >> "$LOG" 2>&1 || echo "$(date) pull FAILED" >> "$LOG"
# don't judge concurrently with the g3+parser chain (same run_tag)
for i in $(seq 1 90); do
  pgrep -f "scratchpad/judge_g3_parser.sh" >/dev/null 2>&1 || break
  echo "$(date) waiting for judge_g3_parser.sh to finish" >> "$LOG"; sleep 60
done
"$PY" -m ops.judge --spec ops/specs/kaya_g1_parser_unlimited_full.yaml --judge-spec gemini-flash >> "$LOG" 2>&1 || echo "$(date) judge FAILED" >> "$LOG"
"$PY" -m ops.build --task all --run-tag g1-parser-full-unlimited >> "$LOG" 2>&1 || echo "$(date) build FAILED" >> "$LOG"
"$PY" -m ops.mine >> "$LOG" 2>&1 || echo "$(date) mine FAILED" >> "$LOG"
"$PY" -m ops.scripts.build_status >> "$LOG" 2>&1 || echo "$(date) build_status FAILED" >> "$LOG"
"$PY" - <<'PYEOF' >> "$LOG" 2>&1
import json, collections
P = "results/cache/g1-parser-full-unlimited/full/G1_oracle_ladder/predictions.jsonl"
c = collections.Counter((json.loads(l).get("status") or "ok") for l in open(P) if l.strip())
print("parser final status:", dict(c), "(target: error=0)")
PYEOF
echo "================ parser-recover watch DONE $(date) ================" >> "$LOG"

#!/usr/bin/env bash
# Detached safety net: guarantees g2 (job 1061344) predictions get pulled off Kaya
# before the Fri 17:00 shutdown, even if the Claude session (and its 12:15 cron)
# dies. Polls sacct every 5 min; once g2 leaves RUNNING/PENDING it pulls once and
# exits. The 12:15 cron re-pulls + judges + rebuilds on top of this (pull is
# idempotent), so this only matters if the cron never fires.
set -uo pipefail
cd /home/lingwei/mpvrdu
PY=envs/mpvrdu/bin/python
LOG=logs/g2_pull_safety.log
echo "================ g2 pull-safety start $(date) ================" >> "$LOG"
for i in $(seq 1 180); do   # up to ~15h at 5-min spacing
  state=$(ssh -o ConnectTimeout=20 kaya "sacct -j 1061344 --format=State -X -n -P 2>/dev/null | head -n1" 2>>"$LOG" | tr -d ' ')
  echo "$(date) g2 state=[$state]" >> "$LOG"
  case "$state" in
    RUNNING|PENDING|"")
      sleep 300 ;;
    *)
      echo "$(date) g2 terminal state=$state -> pulling" >> "$LOG"
      "$PY" -m ops.kaya.kaya pull >> "$LOG" 2>&1 && echo "$(date) pull ok" >> "$LOG" || echo "$(date) pull FAILED" >> "$LOG"
      echo "================ g2 pull-safety DONE $(date) ================" >> "$LOG"
      exit 0 ;;
  esac
done
echo "$(date) g2 pull-safety exhausted loop without terminal state" >> "$LOG"

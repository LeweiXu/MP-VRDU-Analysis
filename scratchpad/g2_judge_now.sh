#!/usr/bin/env bash
set -uo pipefail
cd /home/lingwei/mpvrdu
PY=envs/mpvrdu/bin/python
LOG=logs/g2_judge_now.log
echo "===== g2 judge-now start $(date) =====" >> "$LOG"
"$PY" -m ops.judge --spec ops/specs/kaya_g2_full.yaml --judge-spec gemini-flash >> "$LOG" 2>&1 || echo "judge FAILED" >> "$LOG"
"$PY" -m ops.build --task G2_retrieval --run-tag g2-retrieval-full >> "$LOG" 2>&1 || echo "build FAILED" >> "$LOG"
"$PY" -m ops.mine >> "$LOG" 2>&1 || echo "mine FAILED" >> "$LOG"
"$PY" -m ops.scripts.build_status >> "$LOG" 2>&1 || echo "build_status FAILED" >> "$LOG"
echo "===== g2 judge-now DONE $(date) =====" >> "$LOG"

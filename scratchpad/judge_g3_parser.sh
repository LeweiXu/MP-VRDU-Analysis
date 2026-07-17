#!/usr/bin/env bash
# Detached: judge the two freshly-pulled COMPLETED Kaya arms (g3 hallucination,
# unlimited parser), then rebuild their tables + mine + build_status. No Kaya
# needed (local predictions + gemini API), so it survives VPN drops and session
# death. g2 is handled separately (still running at launch time).
set -uo pipefail
cd /home/lingwei/mpvrdu
PY=envs/mpvrdu/bin/python
LOG=logs/judge_g3_parser.log
echo "================ judge g3+parser start $(date) ================" >> "$LOG"
for spec in kaya_g3_full kaya_g1_parser_unlimited_full; do
  echo "--- judge $spec $(date) ---" >> "$LOG"
  "$PY" -m ops.judge --spec "ops/specs/$spec.yaml" --judge-spec gemini-flash >> "$LOG" 2>&1 \
    && echo "[judge $spec] done $(date)" >> "$LOG" \
    || echo "[judge $spec] FAILED $(date)" >> "$LOG"
done
for tag in g3-hallucination-full g1-parser-full-unlimited; do
  echo "--- build $tag $(date) ---" >> "$LOG"
  "$PY" -m ops.build --task all --run-tag "$tag" >> "$LOG" 2>&1 || echo "build $tag FAILED" >> "$LOG"
done
"$PY" -m ops.mine >> "$LOG" 2>&1 || echo "mine FAILED" >> "$LOG"
"$PY" -m ops.scripts.build_status >> "$LOG" 2>&1 || echo "build_status FAILED" >> "$LOG"
echo "================ judge g3+parser DONE $(date) ================" >> "$LOG"

#!/usr/bin/env bash
set -uo pipefail
cd /home/lingwei/mpvrdu
PY=envs/mpvrdu/bin/python
LOG=logs/parser_rejudge.log
echo "===== parser rejudge start $(date) =====" >> "$LOG"
"$PY" -m ops.judge --spec ops/specs/kaya_g1_parser_unlimited_full.yaml --judge-spec gemini-flash >> "$LOG" 2>&1 || echo "judge FAILED" >> "$LOG"
"$PY" -m ops.build --task all --run-tag g1-parser-full-unlimited >> "$LOG" 2>&1 || echo "build FAILED" >> "$LOG"
"$PY" -m ops.mine >> "$LOG" 2>&1 || echo "mine FAILED" >> "$LOG"
"$PY" -m ops.scripts.build_status >> "$LOG" 2>&1 || echo "build_status FAILED" >> "$LOG"
echo "----- parser.csv n-sum check -----" >> "$LOG"
$PY -c "
import csv
n=sum(int(r['n']) for r in csv.DictReader(open('results/tables/full-g1-parser-full-unlimited/parser.csv')))
print('parser.csv total n =', n, '(want 1585)')
" >> "$LOG" 2>&1
echo "===== parser rejudge DONE $(date) =====" >> "$LOG"

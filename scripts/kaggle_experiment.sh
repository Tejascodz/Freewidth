#!/usr/bin/env bash
# One pre-registered experiment per Kaggle commit (see docs/experiments.md).
#   CONFIG   engine config               (default configs/engine_t4.yaml)
#   DATASET  gsm8k | math500             (default gsm8k)
#   LIMIT    questions                   (default 300)
#   POLICIES comma-separated policies    (default overlap,big_sc2,freewidth)
#   PRIMARY  pre-declared comparisons    (default overlap:big_sc2,overlap:freewidth)
#   OUT      output directory            (default /kaggle/working/bench_<config>_<dataset>)
# Results are written after every policy; the statistics report is built at the end.
set -euo pipefail
START=$(date +%s)
SESSION_HOURS=${SESSION_HOURS:-11.2}   # Kaggle stops commits at 12 h; leave time for the report
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/engine_t4.yaml}
DATASET=${DATASET:-gsm8k}
LIMIT=${LIMIT:-300}
POLICIES=${POLICIES:-overlap,big_sc2,freewidth}
PRIMARY=${PRIMARY:-overlap:big_sc2,overlap:freewidth}
NAME=$(basename "$CONFIG" .yaml)
OUT=${OUT:-/kaggle/working/bench_${NAME}_${DATASET}}
pip install -q -r requirements-gpu.txt
nvidia-smi -L
python -m freewidth calibrate --config "$CONFIG" 2>&1 | grep -E "free width|calibrat|Error"
LEFT=$(python -c "import time; print(round($SESSION_HOURS - (time.time() - $START) / 3600, 2))")
echo "time budget for the benchmark: $LEFT h"
python -m freewidth bench --config "$CONFIG" --dataset "$DATASET" --limit "$LIMIT" \
  --policies "$POLICIES" --out "$OUT" --max-hours "$LEFT" 2>&1 | grep --line-buffered -E "progress|policy done|replica|budget|skipped|Error|Traceback"
cp -r calibration "$OUT/" 2>/dev/null || true
PRIMARY_ARGS=()
[ -n "$PRIMARY" ] && PRIMARY_ARGS=(--primary "$PRIMARY")
python -m freewidth report "$OUT" --out "$OUT/report" --title "$NAME, $DATASET, $LIMIT questions" \
  "${PRIMARY_ARGS[@]}" || echo "report failed; the raw rows are in $OUT/bench_rows.jsonl"

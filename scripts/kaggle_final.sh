#!/usr/bin/env bash
# Kaggle T4 x2, one commit (~8 h for 300 questions). The 300 questions contain the 100 used in
# the earlier runs (same sampler and seed).
#   overlap  - the two-GPU ladder
#   big_sc2  - 4B on BOTH GPUs, 4 samples each, shared QuorumStop (the fair baseline)
#   repair   - Read-Locate-Fork with the 0.4.1 third-party read (the real test of H8)
# The second 4B copy that big_sc2 loads onto GPU 0 is unloaded when big_sc2 finishes.
# Results are written after every policy, so a timeout keeps the finished ones.
set -euo pipefail
START=$(date +%s)
SESSION_HOURS=${SESSION_HOURS:-11.2}   # Kaggle stops commits at 12 h; leave time for the report
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/engine_t4.yaml}
OUT=${OUT:-/kaggle/working/bench_final}
POLICIES=${POLICIES:-overlap,big_sc2,repair}
LIMIT=${LIMIT:-300}
pip install -q -r requirements-gpu.txt
python -m freewidth calibrate --config "$CONFIG" 2>&1 | grep -E "free width|calibrat|Error"
LEFT=$(python -c "import time; print(round($SESSION_HOURS - (time.time() - $START) / 3600, 2))")
echo "time budget for the benchmark: $LEFT h"
python -m freewidth bench --config "$CONFIG" --dataset gsm8k --limit "$LIMIT" \
  --policies "$POLICIES" --out "$OUT" --max-hours "$LEFT" 2>&1 | grep --line-buffered -E "progress|policy done|replica|budget|skipped|Error|Traceback"
cat "$OUT/BENCH.md"
cp -r calibration "$OUT/" 2>/dev/null || true
# primary comparisons declared before the run (README, "Pre-registered experiments", E1)
python -m freewidth report "$OUT" --out "$OUT/report" --title "T4 x2, GSM8K, $LIMIT questions" \
  --primary overlap:big_sc2,overlap:repair || echo "report failed; the raw rows are in $OUT/bench_rows.jsonl"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

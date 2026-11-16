#!/usr/bin/env bash
# Kaggle T4 x2, one commit (~6 h): the two new methods, the earlier best, and the big model alone,
# on the same 100 GSM8K questions and seeds. Results are written straight into /kaggle/working
# after every policy, so a timeout still leaves the finished policies in the Output tab.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/engine_t4.yaml}
OUT=${OUT:-/kaggle/working/bench_4way}
pip install -q -r requirements-gpu.txt
python -m freewidth calibrate --config "$CONFIG" 2>&1 | grep -E "free width|calibrat|Error"
python -m freewidth bench --config "$CONFIG" --dataset gsm8k --limit 100 \
  --policies overlap,repair,freewidth,big_sc --out "$OUT" 2>&1 | grep -E "policy done|Error|Traceback"
cat "$OUT/BENCH.md"
cp -r calibration "$OUT/" 2>/dev/null || true

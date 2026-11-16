#!/usr/bin/env bash
# Rerun Read-Locate-Fork (0.4.1, third-party read) on the same 100 questions (~1.5 h).
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/engine_t4.yaml}
OUT=${OUT:-/kaggle/working/bench_repair}
pip install -q -r requirements-gpu.txt
python -m freewidth calibrate --config "$CONFIG" 2>&1 | grep -E "free width|calibrat|Error"
python -m freewidth bench --config "$CONFIG" --dataset gsm8k --limit 100 --policies repair --out "$OUT" 2>&1 | grep -E "policy done|Error|Traceback"
cat "$OUT/BENCH.md"

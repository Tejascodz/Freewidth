#!/usr/bin/env bash
# A/B on Kaggle T4 x2: freewidth (sequential ladder) vs overlap (two-device ladder),
# same 100 GSM8K questions and seeds as the first run.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/engine_t4.yaml}
pip install -q -r requirements-gpu.txt
python -m freewidth calibrate --config "$CONFIG" 2>&1 | grep -E "free width|calibrat|Error"
python -m freewidth bench --config "$CONFIG" --dataset gsm8k --limit 100 \
  --policies freewidth,overlap --out runs/bench_overlap 2>&1 | grep -E "policy done|Error|Traceback"
cat runs/bench_overlap/BENCH.md
tar czf /kaggle/working/freewidth_overlap.tgz runs calibration

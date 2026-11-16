#!/usr/bin/env bash
# E6 free-width atlas on Kaggle P100 (one GPU). About 1 h; capped by SESSION_HOURS.
set -euo pipefail
cd "$(dirname "$0")/.."
START=$(date +%s)
SESSION_HOURS=${SESSION_HOURS:-1.5}
OUT=${OUT:-/kaggle/working/atlas_p100}
pip install -q -r requirements-gpu.txt
nvidia-smi -L
LEFT=$(python -c "import time; print(round($SESSION_HOURS - (time.time() - $START) / 3600 - 0.1, 2))")
python -m freewidth atlas collect --config configs/atlas_p100.yaml --device cuda:0 --out "$OUT" \
  --max-hours "$LEFT" 2>&1 | grep --line-buffered -E "setting done|failed|budget|free width"
python -m freewidth atlas analyze "$OUT" --out "$OUT/analysis"

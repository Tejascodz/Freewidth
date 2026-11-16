#!/usr/bin/env bash
# E6 free-width atlas on Kaggle T4 x2 (README, Appendix A). Calibration-only: no benchmark questions.
# One process per GPU (no shared Python interpreter), each measuring every other setting.
# About 2-2.5 h; capped by SESSION_HOURS so it cannot exhaust the weekly GPU quota.
set -euo pipefail
cd "$(dirname "$0")/.."
START=$(date +%s)
SESSION_HOURS=${SESSION_HOURS:-3.5}
OUT=${OUT:-/kaggle/working/atlas_t4}
pip install -q -r requirements-gpu.txt bitsandbytes
nvidia-smi -L
LEFT=$(python -c "import time; print(round($SESSION_HOURS - (time.time() - $START) / 3600 - 0.15, 2))")
echo "time budget per worker: $LEFT h"
python -m freewidth atlas collect --config configs/atlas_t4.yaml --device cuda:0 --shard 0/2 \
  --out "$OUT" --max-hours "$LEFT" > "$OUT.gpu0.log" 2>&1 &
P0=$!
sleep 30   # stagger model downloads
python -m freewidth atlas collect --config configs/atlas_t4.yaml --device cuda:1 --shard 1/2 \
  --out "$OUT" --max-hours "$LEFT" > "$OUT.gpu1.log" 2>&1 &
P1=$!
wait $P0 || echo "worker on cuda:0 exited with an error (see $OUT.gpu0.log)"
wait $P1 || echo "worker on cuda:1 exited with an error (see $OUT.gpu1.log)"
grep -hE "setting done|failed|budget" "$OUT".gpu*.log || true
python -m freewidth atlas analyze "$OUT" --out "$OUT/analysis"

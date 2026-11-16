#!/usr/bin/env bash
# Live engine on Kaggle (GPU T4 x2, internet on). From a notebook cell:
#   !cd freewidth && bash scripts/kaggle_engine.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/engine_t4.yaml}

pip install -q -r requirements-gpu.txt
python -m pytest -q

python -m freewidth calibrate --config "$CONFIG"
python -m freewidth solve --config "$CONFIG" "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
python -m freewidth bench --config "$CONFIG" --dataset gsm8k --limit 100 \
  --policies single,sc,hwave,freewidth --out runs/bench_t4

tar czf /kaggle/working/freewidth_engine.tgz runs calibration
echo "saved /kaggle/working/freewidth_engine.tgz"

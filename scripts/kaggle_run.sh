#!/usr/bin/env bash
# Kaggle notebook (GPU T4 x2). In a cell:
#   !git clone https://github.com/<you>/freewidth && cd freewidth && bash scripts/kaggle_run.sh
# Runs the hybrid family on GPU 0 and the dense control on GPU 1 in parallel.
# Kaggle sessions stop at 12 h: everything resumes if you run the script again.
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -q -r requirements-gpu.txt
python -m pytest -q tests

python -m freewidth profile --config configs/kaggle_t4_hybrid.yaml  > profile_hybrid.log 2>&1 &
python -m freewidth profile --config configs/kaggle_t4_dense.yaml   > profile_dense.log  2>&1 &
wait

python -m freewidth bank --config configs/kaggle_t4_hybrid.yaml --max-hours 10.5 > bank_hybrid.log 2>&1 &
python -m freewidth bank --config configs/kaggle_t4_dense.yaml  --max-hours 10.5 > bank_dense.log  2>&1 &
wait

python -m freewidth verify --config configs/kaggle_t4_hybrid.yaml > verify_hybrid.log 2>&1 &
python -m freewidth verify --config configs/kaggle_t4_dense.yaml  > verify_dense.log  2>&1 &
wait

# think-longer variants (optional, long): uncomment when the main banks are done
# python -m freewidth profile --config configs/kaggle_t4_think.yaml
# python -m freewidth bank    --config configs/kaggle_t4_think.yaml --max-hours 10.5

python -m freewidth analyze --config configs/kaggle_t4_hybrid.yaml
python -m freewidth analyze --config configs/kaggle_t4_dense.yaml
python -m freewidth compare runs/t4_hybrid_math500 runs/t4_dense_math500 --out runs/compare.md

# Keep the outputs: Kaggle wipes the working dir when the session ends.
tar czf /kaggle/working/freewidth_runs.tgz runs
echo "saved /kaggle/working/freewidth_runs.tgz"

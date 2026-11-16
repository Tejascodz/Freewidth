#!/usr/bin/env bash
# E1b / H10 (docs/experiments.md): overlap with uncertain-majority accepts switched off, on the same
# 300 GSM8K questions as E1. About 3.5-4 h on Kaggle T4 x2. The session budget is capped at 5 h so a
# slow run cannot use up the weekly GPU quota; finished rows and the report are always written.
# Compare afterwards against E1 (results/t4_gsm8k_E1_2026-11-16):
#   python -m freewidth report results/t4_gsm8k_E1_2026-11-16 <this run> --out <dir>
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=configs/engine_t4_h10.yaml DATASET=gsm8k LIMIT=${LIMIT:-300} POLICIES=overlap PRIMARY="" \
  SESSION_HOURS=${SESSION_HOURS:-5.0} OUT=${OUT:-/kaggle/working/bench_h10} \
  bash scripts/kaggle_experiment.sh

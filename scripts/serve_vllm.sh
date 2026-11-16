#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server on one T4 for the kaggle_t4_vllm config.
#   bash scripts/serve_vllm.sh Qwen/Qwen3-0.6B 0
# T4 (compute capability 7.5) has no bfloat16, hence --dtype half.
set -euo pipefail
MODEL=${1:?model id}
GPU=${2:-0}
CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --dtype half --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --max-num-seqs 128 --port 8000

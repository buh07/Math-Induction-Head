#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="/scratch2/f004ndc/Math Induction Head"
RUN_ROOT="results/phase2/campaign_20260305_025436_gpu01"
LOG_ROOT="logs/phase2/campaign_20260305_025436_gpu01"
cd "$ROOT_DIR"

echo "[$(date -Is)] GPU1 queue start" | tee "$LOG_ROOT/gpu1.queue.log"

echo "[$(date -Is)] Starting gemma_2b phase2 on GPU1" > "$LOG_ROOT/gemma_2b.log"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python scripts/phase2/run_operator_bottleneck_suite.py \
    --model google/gemma-2b \
    --dataset-config configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml \
    --devices 1 \
    --stage full \
    --batch-size 12 \
    --low-cpu-mode \
    --max-cpu-threads 2 \
    --output-root "$RUN_ROOT/gemma_2b" >> "$LOG_ROOT/gemma_2b.log" 2>&1
code=$?
echo "EXIT_CODE=$code" > "$LOG_ROOT/gemma_2b.status"
echo "[$(date -Is)] gemma_2b finished EXIT_CODE=$code" >> "$LOG_ROOT/gemma_2b.log"

echo "[$(date -Is)] Starting gpt2 phase2 on GPU1" > "$LOG_ROOT/gpt2.log"
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python scripts/phase2/run_operator_bottleneck_suite.py \
    --model gpt2 \
    --dataset-config configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml \
    --devices 1 \
    --stage full \
    --batch-size 24 \
    --low-cpu-mode \
    --max-cpu-threads 2 \
    --output-root "$RUN_ROOT/gpt2" >> "$LOG_ROOT/gpt2.log" 2>&1
code=$?
echo "EXIT_CODE=$code" > "$LOG_ROOT/gpt2.status"
echo "[$(date -Is)] gpt2 finished EXIT_CODE=$code" >> "$LOG_ROOT/gpt2.log"

echo "[$(date -Is)] GPU1 queue done" | tee -a "$LOG_ROOT/gpu1.queue.log"

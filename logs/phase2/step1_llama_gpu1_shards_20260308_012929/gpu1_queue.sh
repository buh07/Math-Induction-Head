#!/usr/bin/env bash
set -euo pipefail
cd '/scratch2/f004ndc/Math Induction Head'
echo '[$(date -Is)] GPU 1 queue: subtraction,multiplication' > 'logs/phase2/step1_llama_gpu1_shards_20260308_012929/gpu1.log'
echo '[$(date -Is)] Starting operator=subtraction on GPU=1' > 'logs/phase2/step1_llama_gpu1_shards_20260308_012929/subtraction.log'
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  '.venv/bin/python' scripts/phase2/run_operator_bottleneck_suite.py \
    --model 'meta-llama/Meta-Llama-3-8B' \
    --dataset-config 'configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml' \
    --devices '1' \
    --stage 'full' \
    --batch-size '8' \
    --operators 'subtraction' \
    --operator-shard-mode \
    --low-cpu-mode \
    --max-cpu-threads 2 \
    --output-root 'results/phase2/step1_llama_gpu1_shards_20260308_012929/subtraction' >> 'logs/phase2/step1_llama_gpu1_shards_20260308_012929/subtraction.log' 2>&1
code=$?
echo EXIT_CODE=$code > 'logs/phase2/step1_llama_gpu1_shards_20260308_012929/subtraction.status'
echo '[$(date -Is)] Finished operator=subtraction EXIT_CODE='$code >> 'logs/phase2/step1_llama_gpu1_shards_20260308_012929/subtraction.log'
if [[ $code -ne 0 ]]; then exit $code; fi
echo '[$(date -Is)] Starting operator=multiplication on GPU=1' > 'logs/phase2/step1_llama_gpu1_shards_20260308_012929/multiplication.log'
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  '.venv/bin/python' scripts/phase2/run_operator_bottleneck_suite.py \
    --model 'meta-llama/Meta-Llama-3-8B' \
    --dataset-config 'configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml' \
    --devices '1' \
    --stage 'full' \
    --batch-size '8' \
    --operators 'multiplication' \
    --operator-shard-mode \
    --low-cpu-mode \
    --max-cpu-threads 2 \
    --output-root 'results/phase2/step1_llama_gpu1_shards_20260308_012929/multiplication' >> 'logs/phase2/step1_llama_gpu1_shards_20260308_012929/multiplication.log' 2>&1
code=$?
echo EXIT_CODE=$code > 'logs/phase2/step1_llama_gpu1_shards_20260308_012929/multiplication.status'
echo '[$(date -Is)] Finished operator=multiplication EXIT_CODE='$code >> 'logs/phase2/step1_llama_gpu1_shards_20260308_012929/multiplication.log'
if [[ $code -ne 0 ]]; then exit $code; fi
echo '[$(date -Is)] Queue complete for GPU 1' >> 'logs/phase2/step1_llama_gpu1_shards_20260308_012929/gpu1.log'

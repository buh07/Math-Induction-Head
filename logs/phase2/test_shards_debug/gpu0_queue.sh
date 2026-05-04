#!/usr/bin/env bash
set -euo pipefail
cd '/scratch2/f004ndc/Math Induction Head'
echo '[$(date -Is)] GPU 0 queue: addition,multiplication' > 'logs/phase2/test_shards_debug/gpu0.log'
echo '[$(date -Is)] Starting operator=addition on GPU=0' > 'logs/phase2/test_shards_debug/addition.log'
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  '.venv/bin/python' scripts/phase2/run_operator_bottleneck_suite.py \
    --model 'meta-llama/Meta-Llama-3-8B' \
    --dataset-config 'configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml' \
    --devices '0' \
    --stage 'full' \
    --batch-size '8' \
    --operators 'addition' \
    --operator-shard-mode \
    --low-cpu-mode \
    --max-cpu-threads 2 \
    --output-root 'results/phase2/test_shards_debug/addition' >> 'logs/phase2/test_shards_debug/addition.log' 2>&1
code=$?
echo EXIT_CODE=$code > 'logs/phase2/test_shards_debug/addition.status'
echo '[$(date -Is)] Finished operator=addition EXIT_CODE='$code >> 'logs/phase2/test_shards_debug/addition.log'
if [[ $code -ne 0 ]]; then exit $code; fi
echo '[$(date -Is)] Starting operator=multiplication on GPU=0' > 'logs/phase2/test_shards_debug/multiplication.log'
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  '.venv/bin/python' scripts/phase2/run_operator_bottleneck_suite.py \
    --model 'meta-llama/Meta-Llama-3-8B' \
    --dataset-config 'configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml' \
    --devices '0' \
    --stage 'full' \
    --batch-size '8' \
    --operators 'multiplication' \
    --operator-shard-mode \
    --low-cpu-mode \
    --max-cpu-threads 2 \
    --output-root 'results/phase2/test_shards_debug/multiplication' >> 'logs/phase2/test_shards_debug/multiplication.log' 2>&1
code=$?
echo EXIT_CODE=$code > 'logs/phase2/test_shards_debug/multiplication.status'
echo '[$(date -Is)] Finished operator=multiplication EXIT_CODE='$code >> 'logs/phase2/test_shards_debug/multiplication.log'
if [[ $code -ne 0 ]]; then exit $code; fi
echo '[$(date -Is)] Queue complete for GPU 0' >> 'logs/phase2/test_shards_debug/gpu0.log'

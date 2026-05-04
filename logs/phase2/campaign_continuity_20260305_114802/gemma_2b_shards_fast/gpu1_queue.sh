#!/usr/bin/env bash
set -euo pipefail
cd '/scratch2/f004ndc/Math Induction Head'
echo '[$(date -Is)] GPU 1 queue: addition,multiplication' > 'logs/phase2/campaign_continuity_20260305_114802/gemma_2b_shards_fast/gpu1.log'
echo '[$(date -Is)] Starting operator=addition on GPU=1' > 'logs/phase2/campaign_continuity_20260305_114802/gemma_2b_shards_fast/addition.log'
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  '.venv/bin/python' scripts/phase2/run_operator_bottleneck_suite.py \
    --model 'google/gemma-2b' \
    --dataset-config 'configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml' \
    --devices '1' \
    --stage 'full' \
    --batch-size '12' \
    --operators 'addition' \
    --operator-shard-mode \
    --low-cpu-mode \
    --max-cpu-threads 2 \
    --output-root 'results/phase2/campaign_continuity_20260305_114802/gemma_2b/shards_fast/addition' >> 'logs/phase2/campaign_continuity_20260305_114802/gemma_2b_shards_fast/addition.log' 2>&1
code=$?
echo EXIT_CODE=$code > 'logs/phase2/campaign_continuity_20260305_114802/gemma_2b_shards_fast/addition.status'
echo '[$(date -Is)] Finished operator=addition EXIT_CODE='$code >> 'logs/phase2/campaign_continuity_20260305_114802/gemma_2b_shards_fast/addition.log'
if [[ $code -ne 0 ]]; then exit $code; fi
echo '[$(date -Is)] Starting operator=multiplication on GPU=1' > 'logs/phase2/campaign_continuity_20260305_114802/gemma_2b_shards_fast/multiplication.log'
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  '.venv/bin/python' scripts/phase2/run_operator_bottleneck_suite.py \
    --model 'google/gemma-2b' \
    --dataset-config 'configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml' \
    --devices '1' \
    --stage 'full' \
    --batch-size '12' \
    --operators 'multiplication' \
    --operator-shard-mode \
    --low-cpu-mode \
    --max-cpu-threads 2 \
    --output-root 'results/phase2/campaign_continuity_20260305_114802/gemma_2b/shards_fast/multiplication' >> 'logs/phase2/campaign_continuity_20260305_114802/gemma_2b_shards_fast/multiplication.log' 2>&1
code=$?
echo EXIT_CODE=$code > 'logs/phase2/campaign_continuity_20260305_114802/gemma_2b_shards_fast/multiplication.status'
echo '[$(date -Is)] Finished operator=multiplication EXIT_CODE='$code >> 'logs/phase2/campaign_continuity_20260305_114802/gemma_2b_shards_fast/multiplication.log'
if [[ $code -ne 0 ]]; then exit $code; fi
echo '[$(date -Is)] Queue complete for GPU 1' >> 'logs/phase2/campaign_continuity_20260305_114802/gemma_2b_shards_fast/gpu1.log'

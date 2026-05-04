#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="/scratch2/f004ndc/Math Induction Head"
RUN_ROOT="results/phase2/campaign_20260305_025436_gpu01"
LOG_ROOT="logs/phase2/campaign_20260305_025436_gpu01"
cd "$ROOT_DIR"

echo "[$(date -Is)] GPU0 queue start" | tee "$LOG_ROOT/gpu0.queue.log"

echo "[$(date -Is)] Starting llama3_8b phase2 on GPU0" > "$LOG_ROOT/llama3_8b.log"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python scripts/phase2/run_operator_bottleneck_suite.py \
    --model meta-llama/Meta-Llama-3-8B \
    --dataset-config configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml \
    --devices 0 \
    --stage full \
    --batch-size 8 \
    --low-cpu-mode \
    --max-cpu-threads 2 \
    --output-root "$RUN_ROOT/llama3_8b" >> "$LOG_ROOT/llama3_8b.log" 2>&1
code=$?
echo "EXIT_CODE=$code" > "$LOG_ROOT/llama3_8b.status"
echo "[$(date -Is)] llama3_8b finished EXIT_CODE=$code" >> "$LOG_ROOT/llama3_8b.log"

echo "[$(date -Is)] Starting phase1 rerun on GPU0" > "$LOG_ROOT/phase1_rerun.log"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python scripts/phase1/run_head_validity_suite.py \
    --model meta-llama/Meta-Llama-3-8B \
    --devices 0 \
    --batch-size 8 \
    --seed-list 0,1 \
    --phase3-primary-interventions ablation,amplification \
    --phase3-primary-k-values 5,10 \
    --phase3-primary-scales 0.0,1.25,1.5,2.0 \
    --phase3-multiplicity bh_fdr \
    --phase3-q-max 0.10 \
    --phase3-require-complete-primary-coverage \
    --output-root "$RUN_ROOT/phase1_rerun" >> "$LOG_ROOT/phase1_rerun.log" 2>&1
code=$?
echo "EXIT_CODE=$code" > "$LOG_ROOT/phase1_rerun.status"
echo "[$(date -Is)] phase1 rerun finished EXIT_CODE=$code" >> "$LOG_ROOT/phase1_rerun.log"

echo "[$(date -Is)] GPU0 queue done" | tee -a "$LOG_ROOT/gpu0.queue.log"

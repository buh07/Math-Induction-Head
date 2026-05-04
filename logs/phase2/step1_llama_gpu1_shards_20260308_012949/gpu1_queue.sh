#!/usr/bin/env bash
set -euo pipefail
cd '/scratch2/f004ndc/Math Induction Head'
for op in subtraction multiplication; do
  out_dir="results/phase2/step1_llama_gpu1_shards_20260308_012949/${op}"
  log_file="logs/phase2/step1_llama_gpu1_shards_20260308_012949/${op}.log"
  status_file="logs/phase2/step1_llama_gpu1_shards_20260308_012949/${op}.status"
  echo "[$(date -Is)] starting ${op}" | tee "$log_file"
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    .venv/bin/python scripts/phase2/run_operator_bottleneck_suite.py \
      --model meta-llama/Meta-Llama-3-8B \
      --dataset-config configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml \
      --devices 1 \
      --stage full \
      --batch-size 8 \
      --operators "${op}" \
      --operator-shard-mode \
      --low-cpu-mode \
      --max-cpu-threads 2 \
      --output-root "$out_dir" >> "$log_file" 2>&1
  code=$?
  echo "EXIT_CODE=${code}" > "$status_file"
  echo "[$(date -Is)] finished ${op} EXIT_CODE=${code}" | tee -a "$log_file"
  if [[ $code -ne 0 ]]; then
    exit $code
  fi
done

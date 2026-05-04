#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
SESSION="phase2_full_campaign_${TS}"
RUN_ROOT="results/phase2/campaign_${TS}"
LOG_ROOT="logs/phase2/campaign_${TS}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

GPU_LLAMA="${GPU_LLAMA:-2}"
GPU_GEMMA="${GPU_GEMMA:-3}"
GPU_GPT2="${GPU_GPT2:-4}"
GPU_PHASE1="${GPU_PHASE1:-5}"

launch_phase2_window() {
  local window="$1"
  local model="$2"
  local gpu="$3"
  local batch="$4"
  local out_dir="$5"
  local log_file="$6"
  local status_file="$7"
  tmux new-window -t "$SESSION" -n "$window" \
    "cd '$ROOT_DIR' && \
     echo '[`date -Is`] Starting ${model} on GPU ${gpu}' > '$log_file' && \
     CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
     .venv/bin/python scripts/phase2/run_operator_bottleneck_suite.py \
       --model '${model}' \
       --dataset-config configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml \
       --devices '${gpu}' \
       --stage full \
       --batch-size ${batch} \
       --low-cpu-mode \
       --max-cpu-threads 2 \
       --output-root '${out_dir}' >> '$log_file' 2>&1; \
     code=\$?; echo EXIT_CODE=\$code > '$status_file'; \
     echo '[`date -Is`] Finished ${model} with EXIT_CODE='\$code >> '$log_file'"
}

launch_phase1_window() {
  local window="$1"
  local model="$2"
  local gpu="$3"
  local out_dir="$4"
  local log_file="$5"
  local status_file="$6"
  tmux new-window -t "$SESSION" -n "$window" \
    "cd '$ROOT_DIR' && \
     echo '[`date -Is`] Starting Phase1 rerun on GPU ${gpu}' > '$log_file' && \
     CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
     .venv/bin/python scripts/phase1/run_head_validity_suite.py \
       --model '${model}' \
       --devices '${gpu}' \
       --batch-size 8 \
       --seed-list 0,1 \
       --phase3-primary-interventions ablation,amplification \
       --phase3-primary-k-values 5,10 \
       --phase3-primary-scales 0.0,1.25,1.5,2.0 \
       --phase3-multiplicity bh_fdr \
       --phase3-q-max 0.10 \
       --phase3-require-complete-primary-coverage \
       --output-root '${out_dir}' >> '$log_file' 2>&1; \
     code=\$?; echo EXIT_CODE=\$code > '$status_file'; \
     echo '[`date -Is`] Finished Phase1 rerun with EXIT_CODE='\$code >> '$log_file'"
}

# Start session with first window.
tmux new-session -d -s "$SESSION" -n launcher "cd '$ROOT_DIR' && echo 'Campaign launcher session initialized.'"

launch_phase2_window "llama" "meta-llama/Meta-Llama-3-8B" "$GPU_LLAMA" "8" "$RUN_ROOT/llama3_8b" "$LOG_ROOT/llama3_8b.log" "$LOG_ROOT/llama3_8b.status"
launch_phase2_window "gemma" "google/gemma-2b" "$GPU_GEMMA" "12" "$RUN_ROOT/gemma_2b" "$LOG_ROOT/gemma_2b.log" "$LOG_ROOT/gemma_2b.status"
launch_phase2_window "gpt2" "gpt2" "$GPU_GPT2" "24" "$RUN_ROOT/gpt2" "$LOG_ROOT/gpt2.log" "$LOG_ROOT/gpt2.status"
launch_phase1_window "phase1" "meta-llama/Meta-Llama-3-8B" "$GPU_PHASE1" "$RUN_ROOT/phase1_rerun" "$LOG_ROOT/phase1_rerun.log" "$LOG_ROOT/phase1_rerun.status"

tmux kill-window -t "$SESSION:launcher" >/dev/null 2>&1 || true

cat <<EOF
Launched tmux session: $SESSION
Run root: $RUN_ROOT
Log root: $LOG_ROOT

Attach:
  tmux attach -t $SESSION

Status files:
  $LOG_ROOT/llama3_8b.status
  $LOG_ROOT/gemma_2b.status
  $LOG_ROOT/gpt2.status
  $LOG_ROOT/phase1_rerun.status
EOF

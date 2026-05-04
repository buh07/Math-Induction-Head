#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
SESSION="${SESSION:-phase2_operator_shards_${TS}}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml}"
GPU_LIST="${GPU_LIST:-0,1}"
OPERATORS="${OPERATORS:-addition,subtraction,multiplication}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STAGE="${STAGE:-full}"
RUN_ROOT="${RUN_ROOT:-results/phase2/operator_shards_${TS}}"
LOG_ROOT="${LOG_ROOT:-logs/phase2/operator_shards_${TS}}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
IFS=',' read -r -a OPS <<< "$OPERATORS"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs provided (GPU_LIST)." >&2
  exit 1
fi
if [[ "${#OPS[@]}" -eq 0 ]]; then
  echo "No operators provided (OPERATORS)." >&2
  exit 1
fi

# Round-robin assignment of operators to GPU queues.
declare -A GPU_QUEUES
for gpu in "${GPUS[@]}"; do
  GPU_QUEUES["$gpu"]=""
done
for idx in "${!OPS[@]}"; do
  op="${OPS[$idx]}"
  gpu="${GPUS[$(( idx % ${#GPUS[@]} ))]}"
  if [[ -n "${GPU_QUEUES[$gpu]}" ]]; then
    GPU_QUEUES["$gpu"]+=",${op}"
  else
    GPU_QUEUES["$gpu"]="${op}"
  fi
done

tmux new-session -d -s "$SESSION" -n launcher "cd '$ROOT_DIR' && echo 'Operator shard launcher initialized.'"

for gpu in "${GPUS[@]}"; do
  queue="${GPU_QUEUES[$gpu]}"
  [[ -z "$queue" ]] && continue
  window="gpu${gpu}"
  queue_script="$LOG_ROOT/gpu${gpu}_queue.sh"
  {
    echo "#!/usr/bin/env bash"
    echo "set -euo pipefail"
    echo "cd '$ROOT_DIR'"
    echo "echo '[\$(date -Is)] GPU $gpu queue: $queue' > '$LOG_ROOT/gpu${gpu}.log'"
    IFS=',' read -r -a queue_ops <<< "$queue"
    for op in "${queue_ops[@]}"; do
      out_dir="$RUN_ROOT/${op}"
      log_file="$LOG_ROOT/${op}.log"
      status_file="$LOG_ROOT/${op}.status"
      echo "echo '[\$(date -Is)] Starting operator=$op on GPU=$gpu' > '$log_file'"
      echo "CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \\"
      echo "  '$PYTHON_BIN' scripts/phase2/run_operator_bottleneck_suite.py \\"
      echo "    --model '$MODEL' \\"
      echo "    --dataset-config '$DATASET_CONFIG' \\"
      echo "    --devices '$gpu' \\"
      echo "    --stage '$STAGE' \\"
      echo "    --batch-size '$BATCH_SIZE' \\"
      echo "    --operators '$op' \\"
      echo "    --operator-shard-mode \\"
      echo "    --low-cpu-mode \\"
      echo "    --max-cpu-threads 2 \\"
      echo "    --output-root '$out_dir' >> '$log_file' 2>&1"
      echo "code=\$?"
      echo "echo EXIT_CODE=\$code > '$status_file'"
      echo "echo '[\$(date -Is)] Finished operator=$op EXIT_CODE='\$code >> '$log_file'"
      echo "if [[ \$code -ne 0 ]]; then exit \$code; fi"
    done
    echo "echo '[\$(date -Is)] Queue complete for GPU $gpu' >> '$LOG_ROOT/gpu${gpu}.log'"
  } > "$queue_script"
  chmod +x "$queue_script"
  tmux new-window -t "$SESSION" -n "$window" "bash '$queue_script'"
done

tmux kill-window -t "$SESSION:launcher" >/dev/null 2>&1 || true

SESSION="$SESSION" MODEL="$MODEL" DATASET_CONFIG="$DATASET_CONFIG" GPU_LIST="$GPU_LIST" OPERATORS="$OPERATORS" RUN_ROOT="$RUN_ROOT" LOG_ROOT="$LOG_ROOT" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
payload = {
    "schema_version": "phase2_operator_shard_manifest_v1",
    "session": os.environ["SESSION"],
    "model": os.environ["MODEL"],
    "dataset_config": os.environ["DATASET_CONFIG"],
    "gpus": [item for item in os.environ["GPU_LIST"].split(",") if item],
    "operators": [item for item in os.environ["OPERATORS"].split(",") if item],
    "run_root": str(root),
    "log_root": os.environ["LOG_ROOT"],
}
(root / "shard_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

cat <<EOF
Launched tmux session: $SESSION
Run root: $RUN_ROOT
Log root: $LOG_ROOT
Model: $MODEL
Operators: $OPERATORS
GPU list: $GPU_LIST

Attach:
  tmux attach -t $SESSION

Merge (after all shard .status files show EXIT_CODE=0):
  $PYTHON_BIN scripts/phase2/merge_operator_shards.py \\
    --shard-dirs $(for op in "${OPS[@]}"; do printf "%q " "$RUN_ROOT/$op"; done)\\
    --output-root $RUN_ROOT/merged
EOF

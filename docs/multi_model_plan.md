# Multi-Model Validation Plan

The rebuilt infrastructure allows us to rerun core experiments on multiple LLMs.
This document defines the replication checklist.

## Target Models

- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-3-8b-instruct`
- `microsoft/phi-3-mini-4k-instruct`

Each run must record:
- model name and revision
- tokenizer version
- GPU allocation (GPUs 5 and 6 by default)
- dataset tier hashes used in the run

## Procedure

1. Launch tmux session with `CUDA_VISIBLE_DEVICES=5,6`.
2. For each model:
   - Download weights (if not cached) using `download_model.py` (to be implemented).
   - Execute `python main.py --config configs/<model>.yaml --baseline-tier tier1_in_distribution --baseline-runs 5 --run-sweep --list-tiers`.
   - Store stdout/stderr under `logs/<timestamp>/<model>.log`.
3. After all runs finish, collate manifests in `results/<timestamp>/<model>/run_manifest.json`.

## Acceptance Requirements

- At least two runs per model with distinct seeds.
- Statistical summary (see `src/statistics.py`) computed over the intervention vs. baseline scores for each model.
- Report aggregated in `reports/PUBLICATION_DRAFT.md`.

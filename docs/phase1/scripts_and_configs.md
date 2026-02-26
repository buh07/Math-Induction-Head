# Phase 1 Scripts and Configs (Post-Reorg)

## Canonical script paths

- `scripts/phase1/run_full_experiment.py`
- `scripts/phase1/detect_induction_heads.py`
- `scripts/phase1/run_head_validity_suite.py`
- `scripts/common/summarize_results.py`

Root-level `scripts/*.py` wrappers are kept temporarily for compatibility and print deprecation notices.

## Canonical config paths

- `configs/phase1/experiment_plan.yaml`
- `configs/phase1/experiment_plan_small.yaml`
- `configs/phase1/experiment_plan_multi.yaml`
- `configs/phase1/experiment_plan_gpt2.yaml`
- `configs/phase1/induction_steering.yaml`
- `configs/phase1/induction_steering_llama3.yaml`
- `configs/phase1/induction_blend_llama3.yaml`
- `configs/phase1/induction_gsm_gpt2.yaml`
- `configs/phase1/induction_gsm_llama3.yaml`
- `configs/common/models.yaml`

## Canonical Phase 1 prompt paths

- `prompts/phase1/gsm8k_plain.txt`
- `prompts/phase1/gsm8k_cot.txt`

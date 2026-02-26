# Phase 1 Results Inventory (Old -> New Paths)

This inventory summarizes the result/log path migration performed during the Phase 1 / Phase 2 reorganization.

## Results

- `results/experiments` -> `results/phase1/provisional_pre_fix/experiments`
- `results/experiments_gsm` -> `results/phase1/provisional_pre_fix/experiments_gsm`
- `results/experiments_multi` -> `results/phase1/provisional_pre_fix/experiments_multi`
- `results/experiments_small` -> `results/phase1/provisional_pre_fix/experiments_small`
- `results/head_validity_run_20260224_230046` -> `results/phase1/failed_or_partial/head_validity_run_20260224_230046`
- `results/head_validity_run_20260224_230242` -> `results/phase1/failed_or_partial/head_validity_run_20260224_230242`
- `results/head_validity_run_20260224_230913` -> `results/phase1/failed_or_partial/head_validity_run_20260224_230913`
- `results/head_validity_run_20260225_120553_gpu01` -> `results/phase1/canonical/head_validity_run_20260225_120553_gpu01`
- `results/head_validity_smoke_check_20260224_225940` -> `results/phase1/failed_or_partial/head_validity_smoke_check_20260224_225940`
- `results/head_validity_smoke_gpt2_dev` -> `results/phase1/failed_or_partial/head_validity_smoke_gpt2_dev`
- `results/reruns_20260224_151653` -> `(deleted empty directory)`
- `results/reruns_20260224_151738` -> `results/phase1/{split}/reruns_20260224_151738*`
- `results/reruns_20260224_151738/experiment_plan_main` -> `results/phase1/canonical/reruns_20260224_151738/experiment_plan_main`
- `results/reruns_20260224_151738/experiment_plan_multi` -> `results/phase1/failed_or_partial/reruns_20260224_151738_superseded_or_failed/experiment_plan_multi`
- `results/reruns_20260224_151738/experiment_plan_small` -> `results/phase1/canonical/reruns_20260224_151738/experiment_plan_small`
- `results/reruns_20260224_151738/induction_blend_llama3` -> `results/phase1/canonical/reruns_20260224_151738/induction_blend_llama3`
- `results/reruns_20260224_151738/induction_gsm_gpt2` -> `results/phase1/failed_or_partial/reruns_20260224_151738_superseded_or_failed/induction_gsm_gpt2`
- `results/reruns_20260224_151738/induction_gsm_llama3` -> `results/phase1/failed_or_partial/reruns_20260224_151738_superseded_or_failed/induction_gsm_llama3`
- `results/reruns_20260224_151738/induction_heads_gsm8k_cot.json` -> `results/phase1/canonical/reruns_20260224_151738/induction_heads_gsm8k_cot.json`
- `results/reruns_20260224_151738/induction_heads_gsm8k_plain.json` -> `results/phase1/canonical/reruns_20260224_151738/induction_heads_gsm8k_plain.json`
- `results/reruns_20260224_151738/induction_steering_gpt2` -> `results/phase1/canonical/reruns_20260224_151738/induction_steering_gpt2`
- `results/reruns_20260224_151738/induction_steering_llama3` -> `results/phase1/canonical/reruns_20260224_151738/induction_steering_llama3`
- `results/reruns_failedfix_20260224_163340` -> `results/phase1/canonical/failed_job_reruns_20260224_163340`

## Logs

- `logs/20260219_001426_full_experiment.log` -> `logs/phase1/provisional_pre_fix/20260219_001426_full_experiment.log`
- `logs/20260219_004030_gpt2_experiment.log` -> `logs/phase1/provisional_pre_fix/20260219_004030_gpt2_experiment.log`
- `logs/20260224_230046_head_validity_planA.log` -> `logs/phase1/failed_or_partial/20260224_230046_head_validity_planA.log`
- `logs/20260224_230242_head_validity_planA.log` -> `logs/phase1/failed_or_partial/20260224_230242_head_validity_planA.log`
- `logs/20260224_230913_head_validity_planA.log` -> `logs/phase1/failed_or_partial/20260224_230913_head_validity_planA.log`
- `logs/20260225_120553_head_validity_planA_gpu01.log` -> `logs/phase1/canonical/20260225_120553_head_validity_planA_gpu01.log`
- `logs/20260225_120553_head_validity_planA_gpu01.status` -> `logs/phase1/canonical/20260225_120553_head_validity_planA_gpu01.status`
- `logs/rerun_failed_queue_20260224_163340.log` -> `logs/phase1/canonical/rerun_failed_queue_20260224_163340.log`
- `logs/rerun_queue_20260224_151653.log` -> `logs/phase1/failed_or_partial/rerun_queue_20260224_151653.log`
- `logs/rerun_queue_20260224_151738.log` -> `logs/phase1/canonical/rerun_queue_20260224_151738.log`

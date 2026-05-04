# Phase 2 Remediation Fixes (Schema v2.1 Hardening)

## 1) Context and Scope
This remediation pass implements the Phase 2 hardening plan across orchestration, configs, intervention analysis, CoT pairing, and tests.

Scope of this pass:
- Code + config + tests + documentation (`FIXES.md`)
- No new experiment reruns in this tranche
- Additive schema/interface updates where possible

## 2) Issue-by-Issue Mapping

### Issue A — Multiplicity blocking passed with incomplete primary evidence (High)
- Weakness:
  - Blocking used available q-values only; missing preregistered primary q-rows could be skipped.
- Implemented behavior:
  - Multiplicity now builds expected q-row keys from preregistered fixed-grid target rows.
  - Missing q coverage is explicit and blocking when enabled.
  - Blocking uses worst primary q-value (not best-case), with complete-coverage policy.
- Files changed:
  - `scripts/phase2/run_operator_bottleneck_suite.py`
- Validation tests:
  - `tests/test_phase2_gate_logic.py::test_specificity_gate_multiplicity_requires_complete_primary_q_coverage`
  - `tests/test_phase2_gate_logic.py::test_specificity_gate_multiplicity_requires_all_primary_q_values_below_threshold`

### Issue B — Multiplicity filtering did not enforce preregistered scales (High)
- Weakness:
  - Non-preregistered scales could influence multiplicity blocking.
- Implemented behavior:
  - Multiplicity filtering now applies preregistered `primary_scales` and `primary_k_values`.
  - Only preregistered rows are considered for blocking.
- Files changed:
  - `scripts/phase2/run_operator_bottleneck_suite.py`
  - `src/operator_interventions.py`
- Validation tests:
  - `tests/test_phase2_gate_logic.py::test_specificity_gate_multiplicity_filters_to_primary_scales`
  - `tests/test_phase2_gate_logic.py::test_specificity_gate_primary_k_filter_excludes_non_preregistered_k_rows`

### Issue C — Specificity gate mixed heterogeneous conditions (Medium, academic validity)
- Weakness:
  - Gate decision used aggregated evidence across mixed K/scale conditions.
- Implemented behavior:
  - Specificity now uses `fixed_preregistered_grid` row-wise checks.
  - Gate pass requires all preregistered rows to pass; one failing row fails the gate.
  - Added row-level diagnostics (`row_checks`, `missing_primary_rows`, `missing_non_target_rows`).
- Files changed:
  - `scripts/phase2/run_operator_bottleneck_suite.py`
- Validation tests:
  - `tests/test_phase2_gate_logic.py::test_specificity_gate_does_not_cherry_pick_best_condition_only`
  - `tests/test_phase2_gate_logic.py::test_specificity_gate_fails_when_both_primary_interventions_required_but_missing`

### Issue D — Strict selection/evaluation split failure did not short-circuit pipeline (Medium)
- Weakness:
  - Pipeline could continue into downstream stages after known split leakage.
- Implemented behavior:
  - Split is computed before GPU/model stages.
  - With strict + hard-stop policy, run exits early after datasets with explicit blocked status.
  - Gate summary now reports `phase2_status: blocked_pre_gpu_split_failure` and scope block reasons.
- Files changed:
  - `scripts/phase2/run_operator_bottleneck_suite.py`
- Validation tests:
  - `tests/test_phase2_orchestrator.py::test_phase2_orchestrator_strict_split_failure_hard_stops_pre_gpu`

### Issue E — Family-heldout calibration coverage depended on sampled subset/order (Medium, academic validity)
- Weakness:
  - Heldout nulls sampled only a subset (capped), creating coverage dependence.
- Implemented behavior:
  - Added calibration coverage policy: `exhaustive` (default) vs `sampled`.
  - Exhaustive mode runs all heldout buckets and records coverage metadata.
- Files changed:
  - `scripts/phase2/run_operator_bottleneck_suite.py`
  - `configs/phase2/operator_buckets_llama3.yaml`
  - `configs/phase2/operator_buckets_llama3_full_operators.yaml`
  - `configs/phase2/operator_buckets_llama3.example.yaml`

### Issue F — Primary policy metadata in intervention artifacts was incomplete (Low)
- Weakness:
  - `analysis.primary_set_policy` lacked preregistered scales/K metadata.
- Implemented behavior:
  - Added `primary_scales` and `primary_k_values` in intervention analysis payload.
- Files changed:
  - `src/operator_interventions.py`
- Validation tests:
  - `tests/test_operator_interventions.py::test_run_operator_intervention_sweeps_smoke_with_mocked_eval`

### Issue G — CoT stratification was not weighted by dataset prevalence (Low, academic validity)
- Weakness:
  - Deterministic equal-style stratification could misweight evidence under tight pair budgets.
- Implemented behavior:
  - Added `dataset_pair_allocation` with default `weighted_by_dataset_size`.
  - Weighted quotas use deterministic largest-remainder allocation with seeded tie-breaks.
  - Retained `equal` allocation mode for compatibility.
- Files changed:
  - `src/cot_recruitment.py`
  - `scripts/phase2/run_operator_bottleneck_suite.py`
  - `configs/phase2/operator_buckets_llama3.yaml`
  - `configs/phase2/operator_buckets_llama3_full_operators.yaml`
- Validation tests:
  - `tests/test_cot_recruitment.py::test_build_paired_prompt_examples_weighted_allocation_tracks_dataset_size`
  - `tests/test_cot_recruitment.py::test_build_paired_prompt_examples_weighted_allocation_is_order_robust`
  - `tests/test_cot_recruitment.py::test_build_paired_prompt_examples_equal_allocation_strategy_supported`

## 3) Interface and Config Changes

### Config additions (normalized + persisted)
- `datasets.target_operator_selection_eval_split.hard_stop_on_failure` (default `true`)
- `interventions.primary_k_values` (defaults to localization `k_values`)
- `cot_compare.dataset_pair_allocation` (default `weighted_by_dataset_size`)
- `gates.specificity_condition_policy` (default `fixed_preregistered_grid`)
- `gates.multiplicity_require_complete_primary_coverage` (default `true`)
- `gates.calibration.family_heldout_coverage` (default `exhaustive`)

### Gate summary additive fields
- `schema_revision: "2.1"` retained
- `phases.operator_specificity_gate.condition_policy`
- `phases.operator_specificity_gate.primary_grid` metadata
- `row_checks`, `missing_primary_rows`, `missing_non_target_rows`
- Multiplicity coverage metadata:
  - `n_expected_primary_q_rows`
  - `n_observed_primary_q_rows`
  - `missing_primary_q_rows`
  - `best_primary_q`
  - `worst_primary_q`
- Early-stop status:
  - `overall.phase2_status = blocked_pre_gpu_split_failure`
  - `overall.readiness_block_reasons` includes split scope block reason

### Intervention artifact additive fields
- `analysis.primary_set_policy.primary_scales`
- `analysis.primary_set_policy.primary_k_values`

## 4) Gate Semantics Before/After

- Specificity:
  - Before: mixed-condition aggregation (optimistic under heterogeneity)
  - After: fixed preregistered condition grid, all rows must pass
- Multiplicity:
  - Before: partial q coverage could still pass
  - After: complete preregistered q coverage required (when blocking enabled), with worst-q blocking
- Split leakage:
  - Before: failure could continue into downstream stages
  - After: strict+hard-stop blocks pre-GPU continuation

## 5) Academic Validity Upgrades
- Enforced preregistered condition grid for specificity decisions.
- Explicitly separated split leakage as a hard scope block.
- Upgraded heldout calibration coverage policy to exhaustive-by-default.
- Switched CoT stratification default to prevalence-weighted allocation.
- Added stronger policy metadata to artifacts for post-hoc auditability.

## 6) Tests Added/Updated

Updated tests:
- `tests/test_phase2_gate_logic.py`
  - Added complete-q coverage test
  - Added primary-scale multiplicity filtering test
  - Added primary-K filtering test
  - Updated existing tests for stricter fixed-grid semantics
- `tests/test_phase2_orchestrator.py`
  - Added strict split hard-stop integration test
- `tests/test_operator_interventions.py`
  - Added checks for `primary_scales`/`primary_k_values` in analysis policy
- `tests/test_cot_recruitment.py`
  - Added weighted allocation behavior and order-robustness tests
  - Added equal-allocation compatibility test

Validation command:
- `pytest -q`

## 7) Pending Empirical Validation (Not Done in This Tranche)
These fixes harden logic and artifact semantics, but empirical claims still require reruns:
- Re-run Phase 2 with full preregistered grid and multiplicity blocking enabled.
- Re-run with CoT enabled under weighted allocation and parse-control checks.
- Re-run full-operator config to establish non-target/operator specificity under tightened gates.
- Re-check readiness only after all required gates pass under the hardened policy.

## 8) Legacy Audit + Prereg/Parser/Power Additions (Current Tranche)

### Issue H - Legacy Phase 2 v1 readiness can mislead consumers (artifact-level validity risk)
- Weakness:
  - Legacy `phase2_operator_bottleneck_gate_summary_v1` readiness can be true even when critical gates fail.
- Implemented behavior:
  - Added immutable sidecar audit workflow:
    - `scripts/common/audit_phase2_legacy_artifacts.py`
    - per-run sidecar `legacy_audit.json`
    - aggregate `results/phase2/legacy_audit_index.json`
  - Sidecar reports `audited_ready_for_multimodel` and blocking reasons without mutating legacy gate summaries.
- Files changed:
  - `scripts/common/audit_phase2_legacy_artifacts.py`
  - `results/phase2/operator_bottleneck_run_20260302_222145_gpu2/legacy_audit.json`
  - `results/phase2/legacy_audit_index.json`
- Validation tests:
  - `tests/test_phase2_legacy_audit.py::test_legacy_audit_marks_readiness_false_when_required_gates_fail`
  - `tests/test_phase2_legacy_audit.py::test_legacy_audit_fail_if_v1_missing_sidecar_flag`

### Issue I - Parser fragility needed run-level audit evidence
- Weakness:
  - No standardized artifact quantified default-vs-strict parsing disagreement for Phase 2 runs.
- Implemented behavior:
  - Added parser-audit utilities and run integration:
    - `src/parser_audit.py`
    - `scripts/common/audit_parser_behavior.py`
    - orchestrator emits `parser_audit.json` when intervention prediction samples are present.
- Files changed:
  - `src/parser_audit.py`
  - `scripts/common/audit_parser_behavior.py`
  - `scripts/phase2/run_operator_bottleneck_suite.py`
- Validation tests:
  - `tests/test_parser_audit.py::test_build_parser_audit_report_detects_disagreement_and_accuracy_delta`
  - `tests/test_parser_audit.py::test_collect_parser_audit_samples_from_intervention_runs_reads_prediction_samples`
  - `tests/test_parser_audit.py::test_audit_parser_behavior_cli_writes_output`

### Issue J - Missing prereg/power artifacts for claim hygiene
- Weakness:
  - Phase 2 runs lacked explicit preregistration/power reports in emitted artifacts.
- Implemented behavior:
  - Added prereg + power tooling and run integration:
    - `configs/phase2/preregistration.yaml`
    - `src/power_analysis.py`
    - `scripts/common/power_analysis.py`
    - orchestrator writes `preregistration_used.json` + `power_analysis_report.json`
- Files changed:
  - `configs/phase2/preregistration.yaml`
  - `src/power_analysis.py`
  - `scripts/common/power_analysis.py`
  - `scripts/phase2/run_operator_bottleneck_suite.py`
- Validation tests:
  - `tests/test_power_analysis.py::test_required_n_per_arm_two_proportion_increases_for_smaller_effect`
  - `tests/test_power_analysis.py::test_build_power_analysis_report_uses_primary_grid_and_manifest_counts`
  - `tests/test_power_analysis.py::test_power_analysis_cli_writes_report`
  - `tests/test_phase2_orchestrator.py::test_phase2_orchestrator_datasets_stage_writes_manifest_and_gate_summary`

## 9) Runtime Robustness Fixes from Campaign Bring-Up

### Issue K - Local cache tokenizer integrity could hard-fail full campaign startup
- Weakness:
  - Campaign runs could fail early with opaque downstream tensor-shape errors when tokenizer assets were incomplete/corrupt in local cache.
- Implemented behavior:
  - Added tokenizer health check in model loader.
  - Added tokenizer remote fallback retry when `local_files_only=True` and tokenizer is unusable.
  - Added explicit runtime error if tokenizer remains unusable after fallback.
- Files changed:
  - `src/model_loader.py`
- Validation tests:
  - `tests/test_model_loader.py::test_tokenizer_health_check_flags_empty_tokenization`
  - `tests/test_model_loader.py::test_load_local_model_retries_tokenizer_with_remote_fallback`
  - `tests/test_model_loader.py::test_load_local_model_raises_when_tokenizer_invalid_and_no_fallback`

### Issue L - Localization could crash with empty tokenized batches
- Weakness:
  - Empty `input_ids` could propagate into model forward and fail with cryptic reshape/runtime errors.
- Implemented behavior:
  - Added strict input validation in localization batch preparation:
    - require `input_ids` key,
    - require non-empty sequence length.
  - Emits a direct, actionable tokenizer compatibility error.
- Files changed:
  - `src/arithmetic_localization.py`
- Validation tests:
  - `tests/test_arithmetic_localization.py::test_run_arithmetic_localization_raises_for_empty_tokenized_inputs`

## 10) Phase 2 Performance Upgrade (Batch Auto-Tuning + Operator Sharding)

### Issue M - Fixed batch sizes left GPU utilization on the table
- Weakness:
  - Localization/intervention/CoT stages used a single static batch size, forcing manual tuning and underutilization across models/GPUs.
- Implemented behavior:
  - Added stage-level batch autotuning utility with OOM backoff and bounded retries.
  - Added deterministic baseline-vs-tuned equivalence checks (metric and logit signatures) with hard fallback to baseline on mismatch.
  - Added CLI/config controls for enabling/disabling autotune and tuning bounds.
  - Persisted tuning diagnostics and chosen stage batch sizes in run manifest and gate-summary metadata.
- Files changed:
  - `src/runtime_batch_autotune.py`
  - `scripts/phase2/run_operator_bottleneck_suite.py`
  - `src/experiment_runner.py`
  - `src/operator_interventions.py`
  - `src/cot_recruitment.py`
  - `configs/phase2/operator_buckets_llama3.yaml`
  - `configs/phase2/operator_buckets_llama3_full_operators.yaml`
  - `configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml`
- Validation tests:
  - `tests/test_runtime_batch_autotune.py::test_autotune_oom_backoff_selects_safe_batch`
  - `tests/test_runtime_batch_autotune.py::test_autotune_equivalence_guard_falls_back_to_baseline`
  - `tests/test_experiment_runner.py::test_evaluate_bundle_batch_mode_uses_batched_generation`

### Issue N - Sequential operator execution slowed multi-GPU throughput
- Weakness:
  - Addition/subtraction/multiplication buckets were run together in one process path, limiting simple multi-GPU parallelization.
- Implemented behavior:
  - Added operator-filtered execution (`--operators`) and explicit shard mode (`--operator-shard-mode`) in the Phase 2 orchestrator.
  - Added shard-scope metadata to gate summaries (`scope.operator_coverage`, `scope.is_sharded_run`, `scope.merge_required_for_full_claims`).
  - Added deterministic shard merge utility that validates compatibility and recomputes merged gate summary.
  - Added tmux launcher that assigns operator shards to GPUs round-robin with low-CPU defaults.
- Files changed:
  - `scripts/phase2/run_operator_bottleneck_suite.py`
  - `scripts/phase2/merge_operator_shards.py`
  - `scripts/phase2/launch_operator_shards_tmux.sh`
  - `tests/test_phase2_orchestrator.py`
  - `tests/test_phase2_merge_shards.py`
- Validation tests:
  - `tests/test_phase2_orchestrator.py::test_phase2_orchestrator_operator_filter_limits_dataset_scope`
  - `tests/test_phase2_orchestrator.py::test_phase2_orchestrator_operator_shard_mode_sets_merge_scope_block`
  - `tests/test_phase2_merge_shards.py::test_merge_operator_shards_merges_scope_and_writes_manifests`

# Phase 2 Configs (Operator Heuristic Bottleneck Mainline)

Phase 2 configs live here.

## Current files

- `operator_buckets_llama3.yaml`: runnable Phase 2 default config (addition-only MVP; fastest hardening iteration path).
- `operator_buckets_llama3_full_operators.yaml`: runnable Phase 2 config enabling addition + subtraction + multiplication for cross-operator specificity runs.
- `operator_buckets_llama3_full_operators_campaign.yaml`: full empirical campaign profile (counts/bucket=256, seeds=0/1, multiplicity blocking on).
- `operator_buckets_llama3.example.yaml`: schema/example reference.
- `preregistration.yaml`: preregistered primary comparisons, MESI, multiplicity policy, and planned sample sizes.

Checkpoint note:
- The latest addition MVP run completed with this config family but is currently treated as provisional.
- Phase 2.1 hardening defaults are now encoded in `operator_buckets_llama3.yaml`:
  - calibrated threshold policy (`gates.calibration.*`)
  - CoT-required readiness (`gates.cot_required_for_readiness`)
  - CoT evidence controls (`cot_compare.min_pairs`, `cot_compare.parse_rate_min`, optional CI requirement)
  - localization target-coverage floor (`localization.min_answer_target_valid_rate`)
  - anomaly-blocking controls (`gates.anomaly_blocking.*`)
  - preregistered-primary-set semantics (`interventions.primary_component_set`, `gates.specificity_primary_set`)
  - intervention-signed specificity semantics (`gates.specificity_sign_policy`)
  - preregistered-primary intervention list (`interventions.primary_interventions`)
  - preregistered-primary scales (`interventions.primary_scales`)
  - target-operator selection/evaluation split controls (`datasets.target_operator_selection_eval_split.*`)
  - strict split enforcement option (`datasets.target_operator_selection_eval_split.require_strict`)
  - deterministic stratified CoT pairing controls (`cot_compare.stratify_by_dataset`, `cot_compare.sampling_seed`)
  - family-heldout localizer calibration support (`gates.calibration.localizer_null_policy`, `gates.calibration.max_heldout_buckets`)
  - optional dual-intervention specificity requirement (`gates.specificity_requires_both_primary_interventions`)
  - reporting-only multiplicity policy (`analysis.multiplicity_reporting`)
  - preregistration artifact source (`analysis.preregistration_config`)
  - parser-audit controls (`analysis.parser_audit.*`)
  - batch autotune policy (`runtime.batch_autotune.*`)
  - operator sharding controls (`runtime.operator_filter`, `runtime.operator_shard_mode`)

Use this directory for operator-bucket dataset configs, arithmetic-localizer configs, and cross-operator intervention plans.

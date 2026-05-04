# Phase 2 (Operator Heuristic Bottleneck Mainline)

This directory indexes the post-pivot mainline program.

## Scope

Phase 2 covers:
- operator-specific heuristic bottleneck mapping
- arithmetic-specific causal localization (attention + MLP)
- necessity/sufficiency + cross-operator specificity
- CoT gating/composition experiments
- future multi-model replication of the pivoted program

## Current status

- Documentation and project framing are updated for Phase 2.
- Phase 2 implementation is in place:
  - `scripts/phase2/run_operator_bottleneck_suite.py` (orchestration entrypoint)
  - `configs/phase2/operator_buckets_llama3.yaml` (addition-only default config)
  - `configs/phase2/operator_buckets_llama3_full_operators.yaml` (addition+subtraction+multiplication runnable config)
  - `configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml` (full empirical campaign profile: higher counts + multiplicity blocking)
  - `configs/phase2/preregistration.yaml` (primary comparisons, MESI, planned sample sizes)
  - dataset manifests/diagnostics, localization outputs, intervention outputs, CoT compare outputs, and Phase 2 gate summaries
- CPU-only validation mode remains available via `--scaffold-gpu-stages`.
- Latest addition MVP run (2026-03-03) completed end-to-end with `EXIT_CODE=0` but is treated as **provisional** pending validity hardening:
  - pre-hardening gate semantics in the run's v1 gate summary
  - near-zero localization effect magnitudes in `results/phase2/operator_bottleneck_run_20260302_222145_gpu2/phase2_gate_summary.json`
  - `cot_gating_evidence_gate` unresolved (`status: not_implemented`) in `results/phase2/operator_bottleneck_run_20260302_222145_gpu2/phase2_cot_compare/addition.json`
  - baseline addition accuracy near floor in `results/phase2/operator_bottleneck_run_20260302_222145_gpu2/phase2_interventions/addition_attention_heads.json`
  - immutable legacy sidecar audit marks run not ready: `results/phase2/operator_bottleneck_run_20260302_222145_gpu2/legacy_audit.json`
- Phase 2.1 hardening implementation is now in place (next rerun required for new evidence):
  - gate summary schema upgraded to `phase2_operator_bottleneck_gate_summary_v2` with `schema_revision: 2.1`
  - calibrated thresholds (`derived_thresholds`) and explicit readiness policy (`required_gates_policy`)
  - explicit `scope_warnings`/`scope_blocks` + `overall.readiness_block_reasons`
  - blocking `intervention_sanity_gate` for anomaly/leakage flags
  - forensic anomaly bundle `phase2_intervention_anomaly_report.json` for flagged conditions
  - default Phase 2 config now enables CoT compare for hardening reruns (`cot_compare.enabled: true`)
  - CoT-required readiness enforcement (`cot_required_for_readiness`)
  - substantive CoT evidence thresholds (`cot_compare.effect_abs_min`, `cot_compare.parse_rate_delta_abs_max`, `cot_compare.min_pairs`, `cot_compare.parse_rate_min`)
  - optional CoT CI requirement (`cot_compare.require_accuracy_ci_excludes_zero`)
  - preregistered primary-set checks for specificity (`interventions.primary_component_set`, `gates.specificity_primary_set`)
  - dual-intervention specificity evidence option (`gates.specificity_requires_both_primary_interventions`)
  - intervention-signed specificity semantics (`gates.specificity_sign_policy`) so ablation necessity evidence is scored with correct sign
  - leakage control via target-operator selection/evaluation split (`datasets.target_operator_selection_eval_split`)
  - strict split policy can block readiness when target-operator split cannot be applied (`datasets.target_operator_selection_eval_split.require_strict`)
  - preregistered primary comparison scales for specificity gating (`interventions.primary_scales`)
  - CoT pair sampling now supports deterministic stratification (`cot_compare.stratify_by_dataset`, `cot_compare.sampling_seed`)
  - multiplicity gating uses preregistered-primary q-values and requires all primary comparisons under threshold (`analysis.multiplicity_report.rows[].q_value_primary`)
  - localization calibration policy can combine target-shuffle + family-heldout nulls (`gates.calibration.localizer_null_policy`)
  - localization target-coverage floor (`localization.min_answer_target_valid_rate`)
  - reporting-only multiplicity output (`analysis.multiplicity_report`, BH-FDR by default)
  - stricter arithmetic parsing for operator-bucket evaluation
  - parser-audit output (`parser_audit.json`) built from intervention prediction samples
  - prereg/power artifacts (`preregistration_used.json`, `power_analysis_report.json`)
  - legacy-v1 audit/index tooling (`scripts/common/audit_phase2_legacy_artifacts.py`, `results/phase2/legacy_audit_index.json`)
- See root docs for the current program narrative:
  - `README.md`
  - `overview.md`
  - `ROADMAP.md`
  - `TODO.md`

## Performance upgrade (current)

Phase 2 now supports two runtime-speed features that preserve scientific validity checks:

1) **Safe batch auto-tuning** (per stage)
- runtime config block: `runtime.batch_autotune`
- CLI overrides:
  - `--batch-autotune` / `--no-batch-autotune`
  - `--batch-autotune-min`
  - `--batch-autotune-max`
  - `--batch-autotune-stages localize,intervene,cot`
  - `--batch-equivalence-check` / `--no-batch-equivalence-check`
- emitted in `run_manifest.json`:
  - `tuned_batch_sizes_by_stage`
  - `autotune_probe_history`
  - `equivalence_check_results`

2) **Operator sharding**
- run subset with:
  - `--operators addition,subtraction,multiplication` (any subset)
  - `--operator-shard-mode`
- shard runs are scope-limited:
  - gate summary includes `scope.is_sharded_run=true`
  - readiness is blocked until merge (`scope_blocks` contains `operator_shard_requires_merge` for subset shards)
- merge tool:
  - `scripts/phase2/merge_operator_shards.py`
  - writes merged canonical artifacts:
    - `phase2_gate_summary.json`
    - `run_manifest.json`
    - `merge_manifest.json`

### tmux launcher for operator shards

Use:
- `scripts/phase2/launch_operator_shards_tmux.sh`

Defaults:
- operators: `addition,subtraction,multiplication`
- GPUs: `0,1`
- stage: `full`
- low-CPU mode env (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`)

After shard completion, merge:

```bash
cd '/scratch2/f004ndc/Math Induction Head'
.venv/bin/python scripts/phase2/merge_operator_shards.py \
  --shard-dirs results/phase2/<shard_root>/addition results/phase2/<shard_root>/subtraction results/phase2/<shard_root>/multiplication \
  --output-root results/phase2/<shard_root>/merged
```

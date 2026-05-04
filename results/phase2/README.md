# Phase 2 Results (Operator Heuristic Bottleneck Mainline)

This directory stores Phase 2 run artifacts.

## Current contents

- `operator_bottleneck_run_20260302_222145_gpu2/`
  - legacy addition-only MVP checkpoint (pre-hardening gate semantics)
  - use with caution; decision readiness must come from `legacy_audit.json`
- `legacy_audit_index.json`
  - index of legacy-v1 sidecar audits under `results/phase2/*/legacy_audit.json`

## Trust policy

- Legacy `phase2_operator_bottleneck_gate_summary_v1` runs are immutable historical artifacts.
- Do **not** use raw v1 `overall.ready_for_multimodel_next_tranche` for decisions.
- Use sidecar audit overrides:
  - run-level: `results/phase2/<run_id>/legacy_audit.json`
  - index-level: `results/phase2/legacy_audit_index.json`

## Hardened-run expected artifacts (v2.1+)

A readiness-eligible run should emit at least:
- `phase2_gate_summary.json` (`phase2_operator_bottleneck_gate_summary_v2`, `schema_revision: 2.1`)
- `phase2_localization.json`
- `phase2_interventions.json`
- `phase2_intervention_anomaly_report.json`
- `phase2_cot_recruitment_compare.json`
- `phase2_cross_operator_specificity_matrix.json`
- `phase2_selection_eval_split.json`
- `parser_audit.json`
- `preregistration_used.json`
- `power_analysis_report.json`
- `run_manifest.json`

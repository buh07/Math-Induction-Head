# Phase 2 Scripts (Operator Heuristic Bottleneck Mainline)

Phase 2 scripts live here.

## Current status

- `run_operator_bottleneck_suite.py` is implemented and supports the full Phase 2 stage flow.
- `merge_operator_shards.py` merges per-operator shard runs into canonical merged artifacts.
- `launch_operator_shards_tmux.sh` launches operator shard jobs across GPUs with low-CPU defaults.
- The dataset stage (`--stage datasets`) is CPU-only and writes Phase 2 manifests/diagnostics.
- GPU-backed stages (`localize`, `intervene`, `cross_operator_verify`, `cot_compare`, `full`) are implemented.
- `--scaffold-gpu-stages` is available when you want schema-valid placeholder outputs without model loading.
- Batch auto-tuning is available for GPU stages (`runtime.batch_autotune` + CLI overrides).
- Operator subset execution is available with `--operators`; shard scope can be marked with `--operator-shard-mode`.
- Latest `--stage full` addition MVP run completed, but outputs are provisional pending gate-hardening and CoT-enabled rerun.
- Gate hardening now implemented:
  - gate summary schema `phase2_operator_bottleneck_gate_summary_v2`
  - `schema_revision: 2.1` with explicit `readiness_block_reasons`
  - calibrated thresholds (`derived_thresholds`)
  - localization target-coverage checks
  - balanced primary-set specificity checks across ablation + amplification
  - preregistered primary-set specificity policy
  - reporting-only multiplicity summaries
  - blocking intervention sanity gate
  - CoT-required readiness policy support
  - preregistration/power artifact emission (`preregistration_used.json`, `power_analysis_report.json`)
  - parser-behavior audit emission (`parser_audit.json`) from intervention prediction samples
  - legacy-v1 sidecar audit support via `scripts/common/audit_phase2_legacy_artifacts.py`

## Planned expansion

- arithmetic-specific causal localization suites
- necessity/sufficiency intervention runners
- CoT direct-vs-CoT recruitment comparisons

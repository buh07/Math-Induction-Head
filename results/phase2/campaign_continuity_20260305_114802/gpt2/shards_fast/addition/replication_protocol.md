# Phase 2 Replication Protocol (Operator Heuristic Bottleneck Mainline)

Status: Phase 2 implementation with CPU-first orchestration and GPU-backed localization/intervention/CoT stages.

Recommended sequence:
1. Run `--stage datasets` and validate `dataset_manifest.json` + `dataset_diagnostics.json`.
2. Run `--stage localize` (prefer smoke first) and validate `phase2_localization*.json` + localization gate.
3. Run `--stage intervene` and inspect the specificity matrix + operator-specificity gate.
4. Run `--stage cot_compare` to test CoT gating/composition hypotheses.
5. Use `--stage full` for end-to-end artifacts once CPU/GPU availability allows.
6. Only then mark `ready_for_multimodel_next_tranche=true`.

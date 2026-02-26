# Phase 1 Results (Steering Baseline / Validated Baseline)

This directory contains all retained Phase 1 results, organized by trust status.

## Trust-status directories

- `canonical/` - trusted outputs supporting current Phase 1 conclusions
- `provisional_pre_fix/` - retained historical outputs produced before key hook/parser fixes
- `failed_or_partial/` - failed, interrupted, or smoke runs retained for provenance/debugging

## Canonical anchors

- `canonical/head_validity_run_20260225_120553_gpu01/` (Phase 1 / Plan A validity tranche)
- `canonical/reruns_20260224_151738/` (corrected rerun queue outputs that completed in the main queue)
- `canonical/failed_job_reruns_20260224_163340/` (recovered jobs rerun after fixes)

## Migration map

- `migration_map.json` maps old top-level result paths to their new Phase 1 locations.

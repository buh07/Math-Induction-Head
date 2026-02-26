# Head Validity Replication Protocol (Next Tranche)

Default next replication target: `google/gemma-2b`.

Prerequisites before multi-model replication:
- Phase 2 detector gates pass on Llama-3-8B (non-zero causal metric + positive vs negative separation)
- Phase 3 control steering gate passes on Llama-3-8B (Top-K beats matched-random on at least one criterion)
- Rank stability Spearman >= 0.7 across seeds for Top-50 validated composite ranking

Replication steps (same code path, no redesign):
1. Re-run `scripts/run_head_validity_suite.py` with `--model google/gemma-2b` and same seed list.
2. Keep prompt suites, K values, scales, and gate thresholds identical.
3. Compare gate outcomes, top-head overlap statistics, and control-steering effect sizes.
4. Only after control validity replicates, run arithmetic/GSM extensions.

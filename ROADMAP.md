# Arithmetic Heuristic Bottleneck Research Roadmap

## Executive Summary

This roadmap reflects the post-Plan-A pivot of the project.

Project phase labels used throughout the repository:
- **Phase 1 (Steering Baseline / Validated Baseline)** = completed induction-head steering baseline + reruns + Plan A validity tranche
- **Phase 2 (Operator Heuristic Bottleneck Mainline)** = post-pivot arithmetic bottleneck program (current mainline)

- **Phase 1 (Steering Baseline / Validated Baseline)** is complete enough to serve as the validated comparison axis; Phase 1 / Plan A is the completed induction-head validity tranche.
- The mainline research direction is now **operator-specific heuristic bottlenecks** plus **CoT gating/composition**.
- Future arithmetic improvement claims require arithmetic-specific component identification and causal validation; induction-head amplification is no longer the default primary intervention.

Baseline evidence anchor:
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/gate_summary.json`

## Phase A: Consolidate Validated Baseline (completed Plan A, induction-head validity)

### Goals
- Preserve the Plan A artifact set as the canonical validated baseline tranche.
- Clarify what was validated (control-task targeting/steering) vs what remained weak (arithmetic amplification gains).
- Use Plan A as a comparison/control axis for future arithmetic-specific experiments.

### Tasks
- Freeze and document Plan A artifact interpretation (`gate_summary.json`, Phase 2/3/4 summaries).
- Maintain the head-validity pipeline (`scripts/phase1/run_head_validity_suite.py`) as a regression/validation harness.
- Document the rank-stability caveat as same-set shuffle invariance (not full resampling robustness).

### Exit Criteria
- Baseline artifacts are documented and referenced in README/overview/report.
- Team terminology is aligned on "validated baseline" vs "mainline direction".
- Future experiment plans explicitly treat Plan A as a comparison axis, not the primary arithmetic intervention hypothesis.

### Failure Conditions / Fallback
- Failure: docs continue to overclaim induction-head arithmetic improvement.
- Fallback: block publication-facing summaries until artifact-backed language is restored.

### Artifact Requirements
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/gate_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase2_detector/phase2_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase3_gate_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase4_arithmetic_sanity.json`

## Phase B: Operator-Bucket Dataset Buildout + Error Taxonomy

### Goals
- Build arithmetic datasets that expose operator-specific and subpattern-specific failures.
- Measure failure anatomy (carry/borrow/per-digit errors), not just final-answer accuracy.

### Tasks
- Create bucketed datasets for addition/subtraction/multiplication (no-carry, carry, no-borrow, borrow, etc.).
- Add metadata annotations per prompt (operator, bucket, expected answer, optional per-digit targets).
- Extend evaluation outputs to include per-digit correctness and carry/borrow-specific metrics.

### Exit Criteria
- Dataset buckets exist with reproducible generation configs and hashes.
- Error taxonomy metrics are emitted alongside standard accuracy/parse metrics.
- At least one strong model (Llama-3-8B) shows differentiated performance across buckets.

### Failure Conditions / Fallback
- Failure: bucket definitions do not separate behaviors (all buckets behave identically).
- Fallback: refine bucket definitions and add representation variants (spacing, digit tags, format controls).

### Artifact Requirements
- Dataset manifests/hashes per bucket family
- Diagnostic summaries for bucket distribution and tokenization properties
- Example prompt packs for manual inspection

## Phase C: Arithmetic-Specific Causal Localization (attention + MLP)

### Goals
- Identify arithmetic-relevant components directly from arithmetic tasks.
- Move from induction-proxy ranking to operator-specific causal ranking.

### Tasks
- Implement arithmetic-target causal metrics (answer-token and per-digit target logit deltas).
- Run head and MLP-neuron ablations/patching sweeps by operator bucket.
- Produce operator x component importance matrices with confidence intervals.
- Add robustness checks for rankings (subsample stability, family-heldout stability, seed robustness).

### Exit Criteria
- Localizer outputs non-zero causal effects on arithmetic buckets.
- Ranking beats matched-random controls on at least one operator bucket.
- Robustness metrics pass minimum thresholds, including true subsampling/held-out families.

### Failure Conditions / Fallback
- Failure: arithmetic-specific rankings are unstable or indistinguishable from random.
- Fallback: increase prompt counts, simplify bucket families, or switch to stronger target metrics (e.g., per-digit patching).

### Artifact Requirements
- Localization JSON outputs with schema/version metadata
- Robustness summaries (including explicit "same-set shuffle invariance" vs true subsample metrics)
- Saved candidate component sets with provenance

## Phase D: Necessity/Sufficiency + Cross-Operator Specificity Interventions

### Goals
- Test whether localized operator-specific components are necessary and/or sufficient for performance shifts.
- Quantify operator specificity (addition components should affect addition more than subtraction/multiplication).

### Tasks
- Run ablation (necessity) and amplification/patching (sufficiency) interventions on localized sets.
- Compare against matched-random and baseline induction-head sets.
- Build a cross-operator specificity matrix and report CIs.
- Add rescue experiments where practical (patch correct trajectories into incorrect ones).

### Exit Criteria
- At least one operator-specific component set shows stronger effect on its target operator than on non-target operators.
- Necessity/sufficiency conclusions are artifact-backed and include matched controls.
- Results are stable across at least two seeds or subsamples.

### Failure Conditions / Fallback
- Failure: ablation harms are broad/non-specific and amplification remains null.
- Fallback: interpret as gating/composition bottleneck and proceed to Phase E (CoT recruitment) before more intervention sweeps.

### Artifact Requirements
- Intervention sweep outputs (operator-tagged)
- Cross-operator specificity matrix (machine-readable + summary table)
- Bootstrap CI summaries and effect-size reports

## Phase E: CoT vs Direct-Answer Circuit Recruitment / Gating

### Goals
- Explain why CoT helps math using circuit recruitment and gating/composition differences.
- Distinguish scaffolding effects from arithmetic compute/control effects.

### Tasks
- Run matched direct-answer vs CoT prompt pairs on the same arithmetic instances.
- Compare localized component sensitivity and activation patterns across prompting modes.
- Evaluate whether CoT improves arithmetic via better heuristic gating/composition, not just stronger induction-like signals.
- Add step-level perturbation tests where feasible (format perturbation vs arithmetic correctness).

### Exit Criteria
- Clear, artifact-backed differences between direct-answer and CoT circuit recruitment.
- At least one result supports or falsifies the CoT gating/composition hypothesis.
- Interpretation rules separate formatting effects from arithmetic correctness effects.

### Failure Conditions / Fallback
- Failure: CoT differences reduce to parseability/formatting only.
- Fallback: tighten prompt controls and move to more structured scratchpad prompts.

### Artifact Requirements
- Paired direct-answer/CoT run outputs
- Recruitment/comparison summaries
- Perturbation experiment outputs and interpretation notes

## Phase F: Multi-Model Replication + Publication

### Goals
- Replicate the operator-bottleneck findings across additional models.
- Produce a publication-quality report with clear claims, caveats, and negative results where applicable.

### Tasks
- Replicate Phases C/D (and selected Phase E analyses) on Gemma-2B, then a weaker contrast model.
- Standardize artifacts/schemas across models.
- Draft manuscript/report with a separate Plan A baseline section and pivot rationale.
- Include explicit limitations (tokenization, prompt dependence, stability criteria).

### Exit Criteria
- At least one replication model shows the same directional operator-specific pattern or a clearly interpretable contrast.
- Publication draft contains artifact-backed tables/figures and conservative claims.
- Reproducibility instructions (tmux or scheduler, logs, manifests) are complete.

### Failure Conditions / Fallback
- Failure: cross-model results diverge without explanation.
- Fallback: publish as model-dependent findings with explicit tokenization/architecture hypotheses and targeted follow-up work.

### Artifact Requirements
- Cross-model replication manifests and result summaries
- Final report/manuscript draft
- Reproduction protocol and environment documentation

## Cross-Phase Rules

- Arithmetic amplification claims require arithmetic-specific component identification first.
- Always distinguish "same-set shuffle invariance" from true stability/robustness.
- Separate control-task validity claims from arithmetic-improvement claims.
- Every summary claim must reference concrete artifacts (JSON/logs), not only narrative notes.

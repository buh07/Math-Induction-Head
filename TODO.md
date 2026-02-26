# TODO - Operator Heuristic Bottleneck Program

This tracker reflects the post-Plan-A pivot and repository reorganization.

- **Phase 1 (Steering Baseline / Validated Baseline)** = induction-head steering baseline + reruns + Phase 1 / Plan A validity tranche (completed comparison axis)
- **Phase 2 (Operator Heuristic Bottleneck Mainline)** = active mainline program after the pivot

## Project Direction Update (Post-Plan-A)

- [x] Plan A induction-head validity tranche completed successfully (`EXIT_CODE=0`) and produced a full artifact set.
- [x] Induction-head targeting/steering validity is established on control tasks (hooks active, detector valid, control steering gate passed).
- [x] Arithmetic amplification gains in the Plan A sanity rerun are mostly null/mixed, while top validated-head ablations are strongly harmful.
- [x] Plan A now serves as the **validated baseline / comparison axis** for future mechanistic interventions.
- [x] Project mainline direction pivoted to **operator-specific heuristic bottlenecks** and **CoT gating/composition**.
- [x] Multi-model follow-on is allowed only after preserving the validity discipline used in Plan A.

Canonical Plan A artifacts:
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/gate_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase2_detector/phase2_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase3_gate_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase4_arithmetic_sanity.json`

## Phase 2 (Operator Heuristic Bottleneck Mainline) - Active Program

### Dataset / task buckets (failure anatomy)

- [ ] Build bucketed addition datasets: no-carry, single-carry, cascading-carry, length extrapolation.
- [ ] Build bucketed subtraction datasets: no-borrow, borrow, cascading-borrow, negatives/sign handling.
- [ ] Build bucketed multiplication datasets: table lookup, partial-product composition, carry in partial sums, multi-digit composition.
- [ ] Add prompt metadata schema for `operator`, `bucket`, `expected_answer`, and optional per-digit annotations.
- [ ] Add representation variants (e.g., spaced digits / formatted variants) for bottleneck-disambiguation experiments.

### Arithmetic-specific causal ranking metrics

- [ ] Implement answer-token causal metrics for arithmetic localization (logit/prob delta, KL/L1 changes).
- [ ] Implement per-digit target causal metrics (where per-digit targets are available).
- [ ] Define arithmetic-localizer composite ranking (with matched-random penalties / specificity terms).
- [ ] Add arithmetic-specific causal metric schemas and versioning for outputs.
- [ ] Add bootstrap CI summaries and effect-size reporting for arithmetic-localizer outputs.

### MLP-neuron and head localization

- [ ] Run arithmetic-specific head localization on Llama-3-8B bucketed datasets.
- [ ] Run arithmetic-specific MLP-neuron localization on the same buckets.
- [ ] Produce operator x component importance matrices (attention + MLP).
- [ ] Save localized component sets with provenance (dataset hash, metric config, seed, ranking version).

### Necessity / sufficiency experiments

- [ ] Run ablation (necessity) interventions on operator-localized component sets.
- [ ] Run amplification/patching (sufficiency) interventions on operator-localized component sets.
- [ ] Add matched-random controls and baseline induction-head comparison sets to all operator experiments.
- [ ] Add rescue experiments (correct-run patch into incorrect-run trajectory) for at least one operator bucket.

### Cross-operator specificity matrix

- [ ] Build evaluation matrix: component set (`add`, `sub`, `mul`, random, induction baseline) x operator task bucket.
- [ ] Report ablation deltas, amplification deltas, and CIs for each matrix cell.
- [ ] Define pass/fail criteria for operator specificity (target operator effect > non-target operator effects by CI).

### CoT gating/composition experiments

- [ ] Create matched direct-answer vs CoT prompt pairs for arithmetic bucket tasks.
- [ ] Compare circuit recruitment / sensitivity between direct-answer and CoT runs.
- [ ] Add step-level perturbation tests (format perturbation vs arithmetic correctness) to separate scaffolding from compute.
- [ ] Document whether CoT gains are better explained by gating/composition vs stronger induction-like signals.

### Stability and robustness (required for publication-grade claims)

- [ ] Replace or relabel Phase 2-style rank stability as **same-set shuffle invariance** where prompt content is unchanged.
- [ ] Add true subsample stability for ranking/localization outputs (different prompt subsets, same family).
- [ ] Add family-heldout stability (rank on some families, evaluate on held-out family).
- [ ] Add seed robustness checks for arithmetic-localizer outputs and intervention effects.

## Phase 1 (Steering Baseline / Validated Baseline) - Comparative Controls

These are no longer the mainline arithmetic-improvement track. They remain useful for control experiments and regression checks.

- [ ] Rename/document Phase 2 rank-stability metric as same-set shuffle invariance in generated summaries or post-processing.
- [ ] Add a true subsample/family-heldout stability variant to the induction detector (for comparison to the existing metric).
- [ ] Keep `scripts/phase1/run_head_validity_suite.py` working as a regression harness for hook correctness + control steering validity.
- [ ] Replicate the validated baseline tranche on one additional model (Gemma-2B) after the operator-localizer validity interface is finalized.
- [ ] Maintain `head_sets.json` / detector schema compatibility notes for cross-run comparisons.

## Documentation Cleanup Checklist (Pivot Alignment)

- [x] Create `README.md` as the primary project entrypoint (research + ops hybrid).
- [x] Archive the induction-first overview to `docs/archive/overview_induction_head_hypothesis_20260225.md`.
- [x] Rewrite `overview.md` around operator-specific heuristic bottlenecks + CoT gating/composition.
- [x] Rewrite `ROADMAP.md` to reflect the pivoted phase structure.
- [x] Reorganize `TODO.md` around the new mainline program while preserving historical completed items.
- [x] Update `docs/diagnostics.md` to include arithmetic-specific localization metrics and robustness terminology.
- [x] Update `docs/multi_model_plan.md` to replicate the operator-bottleneck program (not induction-first sweeps).
- [x] Update `reports/phase2/PUBLICATION_DRAFT.md` terminology and section structure (Plan A baseline + pivot rationale).
- [x] Reorganize repo ops/artifacts into explicit `phase1/` and `phase2/` directories for scripts/configs/prompts/results/logs/reports/docs.
- [x] Add Phase 1 trust-status buckets (`canonical`, `provisional_pre_fix`, `failed_or_partial`) and migration maps under `results/phase1/` and `logs/phase1/`.

## Historical Completed Milestones (Preserved from the Rebuild Checklist)

These completed items are preserved for provenance. They represent the successful clean-room rebuild and early experimental infrastructure milestones prior to the pivot.

### Week 1 - Infrastructure Reset (Completed)

- [x] Recreate `src/` with minimal, well-tested utility modules (datasets, logging, configs).
- [x] Configure tooling: formatter, linter, type checker, pytest.
- [x] Implement smoke tests for CLI parsing and dataset generation.
- [x] Draft `CONTRIBUTING.md` describing coding standards and experiment logging rules.

### Weeks 2-3 - Diagnostic Tooling (Completed)

- [x] Design attention-head and neuron-hook APIs plus accompanying unit tests.
- [x] Build staged-ablation runner capable of swapping baselines and suppression modes.
- [x] Add tokenization diagnostics to differentiate single-token vs multi-token arithmetic.
- [x] Document expected metrics and go/no-go thresholds in `docs/diagnostics.md`.

### Weeks 4-5 - Core Experiments (Completed)

- [x] Assemble curated Tiered test suites (in-distribution, near-OOD, symbolic).
- [x] Rebuild evaluation pipelines that log configs, seeds, and dataset hashes.
- [x] Automate parameter sweeps for attention/neuron interventions.
- [x] Capture baseline stability reports before trusting intervention results.

### Week 6+ - Validation & Publication (Completed Milestone Set)

- [x] Replicate promising configurations on multiple model families.
- [x] Extend datasets to GSM8K-style problems and multi-operation arithmetic.
- [x] Produce statistical summaries (bootstrap CIs, effect sizes) with archived manifests.
- [x] Draft publication/report with explicit limitations and audit notes.

## Historical Notes (Superseded by Plan A Validation)

These are retained as context, but they are no longer the active execution driver.

- The prior induction-era validity audit (HF hook targeting and parsing/scoring oversights) was addressed by code fixes and reruns before Plan A.
- Plan A supersedes earlier provisional induction-detection and induction-steering interpretations by validating hooks/detector/controls and rerunning the arithmetic sanity analysis.
- Use the Plan A artifacts listed at the top of this file as the current source of truth for induction-head validity conclusions.

## Standing Rules

- Separate control-task validity claims from arithmetic-improvement claims.
- Use artifact-backed numeric claims only (JSON/log references, not memory).
- Treat induction-head results as a validated baseline / comparison axis unless explicitly running that track.
- Do not call same-set prompt-order invariance "stability" without qualification.
- If a task uncovers missing infrastructure, add it to the active operator-bottleneck backlog before running large sweeps.

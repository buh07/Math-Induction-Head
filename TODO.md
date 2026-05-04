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

## Phase 2.0 Checkpoint (2026-03-03 Addition MVP Run) — Completed, Provisional

- [x] Run completed with `EXIT_CODE=0` (`logs/phase2/20260302_222145_operator_bottleneck_full_gpu2.status`).
- [x] Full pipeline artifacts emitted (`phase2_status = full_pipeline_complete`) in `results/phase2/operator_bottleneck_run_20260302_222145_gpu2/phase2_gate_summary.json`.
- [x] `dataset_bucket_gate`, `localization_validity_gate`, and `operator_specificity_gate` reported pass.
- [x] `cot_gating_evidence_gate` failed because CoT stage was disabled (`status: not_implemented`).
- [x] Baseline addition accuracy is near-floor on the current bucket set (parseable outputs but mostly incorrect).
- [x] This run is marked **provisional** due permissive thresholds and missing CoT evidence.

Why provisional:
- [ ] The 2026-03-03 run used pre-hardening threshold semantics (`localization_nonzero_min: 0.0`, `specificity_ci_low_min: 0.0`) in its emitted v1 gate summary.
- [ ] `results/phase2/operator_bottleneck_run_20260302_222145_gpu2/phase2_gate_summary.json` reports near-zero localization magnitudes for answer-token probability effects.
- [ ] `results/phase2/operator_bottleneck_run_20260302_222145_gpu2/phase2_cot_compare/addition.json` shows CoT stage disabled (`not_implemented`).
- [ ] Current outputs are **not sufficient for mechanistic claim** without hardened gate criteria and CoT evidence.

Issues observed in the most recent run (must be resolved before Phase 2.2):
- [ ] [BLOCKER][ANALYSIS] Localization signal is effectively silent despite reported pass (`effect_nonzero_rate_max = 0.0` for both attention and MLP checks).
- [ ] [BLOCKER][ANALYSIS] Answer-token probability effects are numerically tiny (`~3.5e-07` max for attention heads; `~6.46e-08` max for MLP neurons), so current ranking may not be causally meaningful.
- [ ] [BLOCKER][VALIDITY] MLP robustness metrics are missing in the gate summary (`same_set_shuffle_invariance`, `subsample_stability`, `family_heldout_stability`, `seed_robustness` all null), but the run still passed localization validity.
- [ ] [BLOCKER][VALIDITY] CoT gate did not run meaningfully (`status: not_implemented`), so CoT gating/composition claims are untested for this tranche.
- [ ] [BLOCKER][ANALYSIS] Baseline addition performance is near-floor with full parseability (`0.0`, `0.0625`, `0.0` accuracy across addition buckets; `parse_rate = 1.0`), indicating severe correctness failure not addressed by current gate semantics.
- [ ] [BLOCKER][VALIDITY] Attention intervention behavior includes extreme shifts (e.g., `K5:bottom` ablation yielding near-perfect accuracy), requiring dedicated leakage/sanity analysis before accepting operator-specific conclusions.

## Phase 2.1 Validity Hardening Tranche — Active (Blocking)

### Hardening fixes to implement now

- [x] [BLOCKER][VALIDITY] Replace permissive gate defaults with calibrated hardening thresholds; fail on numerically negligible localization effects instead of passing at zero.
- [x] [BLOCKER][VALIDITY] Enforce CoT-required readiness semantics so `ready_for_multimodel_next_tranche` is always false when CoT gate is disabled/not implemented.
- [x] [BLOCKER][VALIDITY] Fail localization validity when requested robustness modes are missing/null for any enabled component family (attention + MLP).
- [x] [BLOCKER][VALIDITY] Fail operator-specificity validity when run scope lacks non-target operators and policy requires cross-operator evidence.
- [x] [BLOCKER][VALIDITY] Add anomaly-blocking sanity gate for extreme intervention jumps before accepting specificity conclusions.
- [x] [BLOCKER][ANALYSIS] Add forensic output fields (`prediction_samples` with generated text + parsed + target) to intervention artifacts for anomaly triage.
- [x] [BLOCKER][ANALYSIS] Harden numeric parsing/evaluation to avoid malformed-output parse artifacts inflating arithmetic conclusions.
- [x] [BLOCKER][VALIDITY] Enforce preregistered primary-set specificity policy (`interventions.primary_component_set`, `gates.specificity_primary_set`) in gate decisions and intervention analysis outputs.
- [x] [BLOCKER][VALIDITY] Require balanced primary-set evidence across both ablation and amplification (`gates.specificity_requires_both_primary_interventions`) for hardened specificity claims.
- [x] [BLOCKER][VALIDITY] Fix specificity sign semantics so ablation is evaluated as necessity-style harm (signed policy) rather than requiring positive deltas.
- [x] [BLOCKER][VALIDITY] Add selection/evaluation split for target-operator datasets to reduce localization/intervention leakage in target-operator claims.
- [x] [BLOCKER][ANALYSIS] Add reporting-only multiplicity output (`analysis.multiplicity_report`, BH-FDR default) to intervention artifacts.
- [x] [BLOCKER][VALIDITY] Ensure multiplicity blocking uses preregistered-primary comparisons (`q_value_primary`) instead of all intervention rows.
- [x] [BLOCKER][VALIDITY] Add optional multiplicity-based gate blocking policy (`gates.multiplicity_blocking.*`) for stricter reruns.
- [x] [BLOCKER][OPS] Add full-operator runnable config (`configs/phase2/operator_buckets_llama3_full_operators.yaml`) while keeping addition-only default for fast iteration.
- [x] [BLOCKER][VALIDITY] Add localization answer-target coverage floor (`localization.min_answer_target_valid_rate`) to avoid low-coverage tokenization artifacts.
- [x] [BLOCKER][VALIDITY] Strengthen CoT gate with paired-count minimum, parse-rate floor, and optional CI-excludes-zero policy.
- [x] [BLOCKER][ANALYSIS] Make CoT paired-sampling deterministic and stratified by dataset to remove order-dependent pair selection artifacts.
- [x] [BLOCKER][VALIDITY] Extend localizer calibration policy to support family-heldout null controls in addition to target shuffling.
- [x] [BLOCKER][OPS] Add immutable legacy-v1 audit sidecars (`legacy_audit.json`) plus index (`results/phase2/legacy_audit_index.json`) so provisional v1 readiness cannot be consumed directly.
- [x] [BLOCKER][ANALYSIS] Add parser-audit artifact (`parser_audit.json`) with default-vs-strict parse agreement and ambiguity/adjudication output.
- [x] [BLOCKER][ANALYSIS] Add preregistration + power-analysis artifacts (`preregistration_used.json`, `power_analysis_report.json`) as Phase 2 run prerequisites.

Implementation status note:
- [x] Code hardening implemented across orchestrator/config/localization/intervention/evaluation modules and validated with full local test suite.
- [ ] Scientific validation still pending until a new hardening rerun is executed and artifacts are reviewed.

### A) Gate semantics hardening (first priority)

- [x] [BLOCKER][VALIDITY] Raise `localization_nonzero_min` from `0.0` to a non-trivial threshold and document rationale.
- [x] [BLOCKER][VALIDITY] Raise `specificity_ci_low_min` from `0.0` to a non-trivial threshold and document rationale.
- [x] [BLOCKER][VALIDITY] Update readiness logic so `ready_for_multimodel_next_tranche` cannot be true when required gates are missing/disabled.
- [x] [BLOCKER][VALIDITY] Define required-gate policy for Phase 2 addition MVP (CoT required vs optional) and encode in TODO acceptance criteria.

### B) Localization validity tightening

- [x] [BLOCKER][VALIDITY] Require non-negligible effect size checks (not just non-empty metrics).
- [x] [BLOCKER][VALIDITY] Require robustness for all enabled component types (attention + MLP), or explicitly mark missing robustness as fail.
- [x] [BLOCKER][ANALYSIS] Distinguish `same-set shuffle invariance` from true robustness in all Phase 2 summaries.

### C) Intervention sanity and leakage checks

- [x] [BLOCKER][VALIDITY] Add a dedicated sanity pass that compares baseline predictions vs intervention predictions on identical prompts for impossible jumps/regressions.
- [x] [BLOCKER][VALIDITY] Verify intervention directionality consistency (ablation vs amplification) against localization ranking expectations.
- [x] [BLOCKER][ANALYSIS] Emit `phase2_intervention_anomaly_report.json` with flagged conditions and capped `prediction_samples` for forensic review.
- [ ] [BLOCKER][ANALYSIS] Investigate and explain extreme condition deltas before accepting specificity conclusions.

### D) CoT gate completion

- [x] [BLOCKER][VALIDITY] Enforce CoT-required readiness semantics and block readiness when CoT gate is unresolved.
- [x] [BLOCKER][VALIDITY] Enable CoT compare in config for the next hardening run.
- [ ] [BLOCKER][ANALYSIS] Require paired direct-vs-CoT artifact and gate result before unblocking Phase 2.2.
- [x] [BLOCKER][OPS] Add explicit failure-handling behavior in gate summary when CoT remains disabled/not_implemented.

### E) Rerun protocol (addition-only hardening rerun)

- [ ] [BLOCKER][OPS] Execute hardening rerun on addition-only buckets with updated gates.
- [ ] [BLOCKER][OPS] Run at least two seeds and record seed-level deltas in artifact summary.
- [ ] [BLOCKER][ANALYSIS] Publish post-rerun verdict: pass/fail on each gate with exact artifact references.

### F) Exit criteria for Phase 2.1

- [ ] Non-trivial localization effects + robustness present for required component modes.
- [ ] Specificity evidence survives tightened CI thresholds.
- [ ] CoT gate is executed and resolved per policy.
- [ ] Only then transition to Phase 2.2 expansion.

### G) Acceptance checklist for next hardening rerun (artifact anchored)

- [ ] `results/phase2/<run_id>/phase2_gate_summary.json` uses `phase2_operator_bottleneck_gate_summary_v2` and includes `derived_thresholds`, `required_gates_policy`, `scope_warnings`, and `intervention_sanity_gate`.
- [ ] `results/phase2/<run_id>/run_manifest.json` records calibrated thresholds and calibration config used for this run.
- [ ] `results/phase2/<run_id>/phase2_localization.json` shows non-negligible localized effects above hardened floors and non-null robustness for required modes.
- [ ] `results/phase2/<run_id>/phase2_interventions/*.json` include `sanity_flags` and `prediction_samples` for flagged conditions.
- [ ] `results/phase2/<run_id>/phase2_intervention_anomaly_report.json` exists and summarizes flagged conditions/datasets with forensic samples.
- [ ] `results/phase2/<run_id>/parser_audit.json` reports parse-mode agreement, ambiguity rate, and adjudication samples.
- [ ] `results/phase2/<run_id>/preregistration_used.json` and `results/phase2/<run_id>/power_analysis_report.json` exist and match the run's preregistered primary grid.
- [ ] `results/phase2/<run_id>/phase2_cot_compare/*.json` are `status: ok` (or explicitly policy-failed) and the corresponding `cot_gating_evidence_gate` outcome is reflected in readiness logic.
- [ ] `results/phase2/<run_id>/phase2_cross_operator_specificity_matrix.json` includes non-target operator columns before any operator-specificity pass claim is accepted.

## Phase 2.2 Expansion Program — Deferred (Blocked by Phase 2.1 gates)

- [ ] [DEFERRED] Do not start until all Phase 2.1 blockers pass.

### Dataset / task buckets (failure anatomy)

- [ ] [DEFERRED] Build bucketed addition datasets: no-carry, single-carry, cascading-carry, length extrapolation.
- [ ] [DEFERRED] Build bucketed subtraction datasets: no-borrow, single-borrow, cascading-borrow, negatives/sign handling.
- [ ] [DEFERRED] Build bucketed multiplication datasets: table lookup, partial-product composition, carry in partial sums, multi-digit composition.
- [ ] [DEFERRED] Add prompt metadata schema for `operator`, `bucket`, `expected_answer`, and optional per-digit annotations.
- [ ] [DEFERRED] Add representation variants (e.g., spaced digits / formatted variants) for bottleneck-disambiguation experiments.

### Arithmetic-specific causal ranking metrics

- [ ] [DEFERRED] Implement answer-token causal metrics for arithmetic localization (logit/prob delta, KL/L1 changes).
- [ ] [DEFERRED] Implement per-digit target causal metrics (where per-digit targets are available).
- [ ] [DEFERRED] Define arithmetic-localizer composite ranking (with matched-random penalties / specificity terms).
- [ ] [DEFERRED] Add arithmetic-specific causal metric schemas and versioning for outputs.
- [ ] [DEFERRED] Add bootstrap CI summaries and effect-size reporting for arithmetic-localizer outputs.

### MLP-neuron and head localization

- [ ] [DEFERRED] Run arithmetic-specific head localization on Llama-3-8B bucketed datasets.
- [ ] [DEFERRED] Run arithmetic-specific MLP-neuron localization on the same buckets.
- [ ] [DEFERRED] Produce operator x component importance matrices (attention + MLP).
- [ ] [DEFERRED] Save localized component sets with provenance (dataset hash, metric config, seed, ranking version).

### Necessity / sufficiency experiments

- [ ] [DEFERRED] Run ablation (necessity) interventions on operator-localized component sets.
- [ ] [DEFERRED] Run amplification/patching (sufficiency) interventions on operator-localized component sets.
- [ ] [DEFERRED] Add matched-random controls and baseline induction-head comparison sets to all operator experiments.
- [ ] [DEFERRED] Add rescue experiments (correct-run patch into incorrect-run trajectory) for at least one operator bucket.

### Cross-operator specificity matrix

- [ ] [DEFERRED] Build evaluation matrix: component set (`add`, `sub`, `mul`, random, induction baseline) x operator task bucket.
- [ ] [DEFERRED] Report ablation deltas, amplification deltas, and CIs for each matrix cell.
- [ ] [DEFERRED] Define pass/fail criteria for operator specificity (target operator effect > non-target operator effects by CI).

### CoT gating/composition experiments

- [ ] [DEFERRED] Create matched direct-answer vs CoT prompt pairs for arithmetic bucket tasks.
- [ ] [DEFERRED] Compare circuit recruitment / sensitivity between direct-answer and CoT runs.
- [ ] [DEFERRED] Add step-level perturbation tests (format perturbation vs arithmetic correctness) to separate scaffolding from compute.
- [ ] [DEFERRED] Document whether CoT gains are better explained by gating/composition vs stronger induction-like signals.

### Stability and robustness (required for publication-grade claims)

- [ ] [DEFERRED] Replace or relabel rank stability as **same-set shuffle invariance** where prompt content is unchanged.
- [ ] [DEFERRED] Add true subsample stability for ranking/localization outputs (different prompt subsets, same family).
- [ ] [DEFERRED] Add family-heldout stability (rank on some families, evaluate on held-out family).
- [ ] [DEFERRED] Add seed robustness checks for arithmetic-localizer outputs and intervention effects.

## Phase 1 (Steering Baseline / Validated Baseline) — Historical / Comparative

These tasks remain comparative controls and regression checks, not the active arithmetic-improvement path.

- [ ] Phase 1 is **not** an active optimization path for arithmetic performance claims in Phase 2 mainline work.
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

## Current Critical Unknowns (Must Resolve in Phase 2.1)

- [ ] Whether localization signal magnitudes are causally meaningful vs numerically negligible.
- [ ] Whether large intervention effects reflect valid operator circuits vs confounding behavior.
- [ ] Whether CoT gating/composition evidence changes interpretation materially.
- [ ] Whether current readiness flag semantics are over-admitting progress.

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

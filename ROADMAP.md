# Induction Head Research Roadmap (Fresh Start)

## Executive Summary

This roadmap assumes a complete rebuild of the mechanistic-interpretability workflow.
No previous experiments or scripts are considered trustworthy; every component must be
recreated and audited before collecting new data. The plan spans four phases that
progress from infrastructure to publication, with explicit decision gates and success
criteria at each step.

**Timeline:** ~6 weeks | **GPU Budget:** ≤25 hours | **Person-Hours:** ~35

---

## Phase 0 – Infrastructure Reset (Week 1)

**Goals**
- Recreate the repository layout with minimal, well-documented scaffolding.
- Stand up coding standards (formatter, linter, type checker) and automated tests.
- Specify configuration formats for datasets, models, and interventions.

**Tasks**
1. Implement lightweight utility modules for dataset generation, logging, and config loading.
2. Add smoke tests that exercise data loaders and CLI argument parsing.
3. Document coding conventions and experiment-tracking requirements in `CONTRIBUTING.md`.

**Exit Criteria**
- Utilities tested locally (unit tests pass).
- `main.py` runs end-to-end in “dry-run” mode without contacting external services.
- Decision memo written that confirms readiness for diagnostic tooling.

---

## Phase 1 – Diagnostic Tooling (Weeks 2–3)

**Goals**
- Build modular hooks for attention-head and MLP-neuron interventions.
- Implement tokenization diagnostics to separate single-token from multi-token issues.
- Define quantitative gatekeeping metrics (entropy deltas, accuracy deltas, etc.).

**Tasks**
1. Implement staged-ablation runners with configurable layer/Head selections.
2. Create multi-metric measurement scripts (entropy, behavioral impact, logit attribution).
3. Add tokenization stress tests that scale operand ranges and operation types.

**Exit Criteria**
- Diagnostics produce reproducible metrics on synthetic prompts.
- Hook implementations include automated tests plus sanity checks (e.g., zeroing out heads).
- Steering/ablation configs validated via dry runs and documented in `docs/diagnostics.md`.

---

## Phase 2 – Core Experiments (Weeks 4–5)

**Goals**
- Collect brand-new baselines on curated arithmetic and symbolic datasets.
- Run attention and neuron interventions only after baselines stabilize.
- Capture statistical analyses with preregistered acceptance thresholds.

**Tasks**
1. Assemble Tiered test suites (in-distribution arithmetic, near-OOD numbers, symbolic patterns).
2. Execute experiments with automated sweeps over suppression/amplification parameters.
3. Store all artifacts (configs, logs, plots) in timestamped directories with README files.

**Exit Criteria**
- Baseline variance <3 percentage points across three independent seeds.
- Intervention runs logged with complete metadata (model version, dataset hash, git commit).
- Scenario decision (Improvement / Null / Mixed) documented with supporting plots.

---

## Phase 3 – Extension & Publication (Week 6+)

**Goals**
- Validate findings across additional models and harder benchmarks (e.g., GSM8K).
- Produce visualizations that explain head/neuron roles.
- Draft a manuscript or technical report with transparent limitations.

**Tasks**
1. Replicate top-performing configurations on at least two alternative model families.
2. Extend datasets to include multi-step reasoning and multi-operation arithmetic.
3. Run statistical validation scripts (bootstrap CIs, non-parametric tests) and archive outputs.
4. Write the publication draft plus an audit appendix summarizing every run.

**Exit Criteria**
- Cross-model validation logs available and internally reviewed.
- Visual assets (attention maps, accuracy curves) linked to raw data.
- Draft ready for internal circulation with clearly labeled open questions.

---

## Contingency Handling

- **Tooling failure:** roll back to Phase 0, fix utilities, and rerun smoke tests before attempting diagnostics.
- **Unstable baselines:** halt interventions, increase dataset sizes, and rerun until variance shrinks.
- **Ambiguous results:** collect 2× additional problems per tier before making scenario calls.
- **Resource overruns:** prioritize Tier 1 datasets and defer cross-model work until baseline confidence restores.

---

## Documentation Requirements

1. Every script must emit a run manifest (JSON or YAML) summarizing parameters and git commit.
2. Logs should be rotated per run and stored under `logs/YYYYMMDD_HHMM/`.
3. Experiment notes (rationale, anomalies, follow-ups) belong in `notes/` with timestamped files.
4. No result is deemed complete until another team member can replay it using only the manifest.

---

## Next Actions

1. Implement the Phase 0 utilities and smoke tests.
2. Draft diagnostics specifications (APIs, expected metrics) for review.
3. Schedule the first audit checkpoint after Phase 1 to confirm readiness for expensive runs.

# TODO – Fresh Execution Plan

This checklist tracks the clean-room rebuild of the induction-head project. Nothing
from the previous implementation is trusted; only tasks listed here should drive the
next iteration.

## Week 1 – Infrastructure Reset

- [x] Recreate `src/` with minimal, well-tested utility modules (datasets, logging, configs).
- [x] Configure tooling: formatter, linter, type checker, pytest.
- [x] Implement smoke tests for CLI parsing and dataset generation.
- [x] Draft `CONTRIBUTING.md` describing coding standards and experiment logging rules.

## Weeks 2–3 – Diagnostic Tooling

- [x] Design attention-head and neuron-hook APIs plus accompanying unit tests.
- [x] Build staged-ablation runner capable of swapping baselines and suppression modes.
- [x] Add tokenization diagnostics to differentiate single-token vs multi-token arithmetic.
- [x] Document expected metrics and go/no-go thresholds in `docs/diagnostics.md`.

## Weeks 4–5 – Core Experiments

- [x] Assemble curated Tiered test suites (in-distribution, near-OOD, symbolic).
- [x] Rebuild evaluation pipelines that log configs, seeds, and dataset hashes.
- [x] Automate parameter sweeps for attention/neuron interventions.
- [x] Capture baseline stability reports before trusting intervention results.

## Week 6+ – Validation & Publication

- [x] Replicate promising configurations on multiple model families.
- [x] Extend datasets to GSM8K-style problems and multi-operation arithmetic.
- [x] Produce statistical summaries (bootstrap CIs, effect sizes) with archived manifests.
- [x] Draft publication/report with explicit limitations and audit notes.

## Experiment Execution (Next)

- [ ] Part A: Run `scripts/run_full_experiment.py` with the baseline plan to collect results for all tiers.
- [ ] Part B: Review `attention_suppression` sweep outputs and identify promising scales per model.
- [ ] Part C: Review `neuron_suppression` sweep outputs and compare against attention sweeps.
- [ ] Part D: Feed collected scores into `scripts/summarize_results.py` and update `reports/PUBLICATION_DRAFT.md`.

## Induction-Head Validation (New)

- [ ] Implement induction-head detection metrics (attention entropy, previous-token matching, logit influence) and produce `results/induction_heads_<model>.json`.
- [x] Extend hook configs so we can scale specific `(layer, head)` pairs (induction vs. non-induction) rather than whole layers, including optional downscaling for competing heads.
- [ ] Create a targeted steering plan (`configs/induction_steering.yaml`) that mirrors the original 0.9× suppression experiment (baseline, mild suppression, strong suppression, amplification).
- [ ] Run the targeted experiments on the original model (Llama-2-7B or Llama-3-8B) and record accuracy improvements with manifests.
- [ ] Use `scripts/summarize_results.py` to generate bootstrap confidence intervals / effect sizes for baseline vs. forced-induction runs.

## Next Research Directions

- [ ] Design layer-wise custom attention steering vectors that ablate arithmetic heuristics while preserving generic context handling; test whether full-stack steering improves accuracy across Tier 1–Tier 4.
- [ ] Build an induction-head amplification protocol that boosts identified heads (with optional mild suppression of others) and measure resulting performance shifts.
- [ ] Explore blended runs that combine steering-vector ablations with induction-head amplification to map the interaction surface between the two interventions.

## Standing Rules

- Do not reuse legacy notebooks, cached results, or partially implemented hooks.
- Every new script must emit a run manifest and save logs under `logs/<timestamp>/`.
- If a task uncovers missing infrastructure, pause and add it to Week 1 backlog before proceeding.

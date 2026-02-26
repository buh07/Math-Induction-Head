# Diagnostics, Metrics, and Gates

This document defines the diagnostics required before interpreting expensive arithmetic experiments.

Repository phase context:
- **Phase 1 (Steering Baseline / Validated Baseline)** established the validity-first gating discipline on induction-head control tasks.
- **Phase 2 (Operator Heuristic Bottleneck Mainline)** reuses that discipline for arithmetic-specific localization and intervention claims.

The same gating philosophy used in Phase 1 / Plan A (instrumentation validity -> detector/localizer validity -> control specificity -> arithmetic interpretation) remains the standard for the pivoted operator-bottleneck program.

## Instrumentation Primitives (Attention + Neuron Hooks)

Attention and neuron hooks are instrumentation primitives for many experiment families. They are not induction-only machinery.

### Required hook diagnostics

- **Hook configuration provenance**
  - Every hook set must record where it came from (detector/localizer output, manual list, random control seed).
- **Hook application counters / debug traces**
  - Use debug counters where available to confirm targeted hooks actually fired on the intended layers.
- **No-op detection**
  - Head/neuron-targeted hooks must fail loudly in strict validation modes if the tensor cannot be mapped to the intended axes.
- **Synthetic tensor regression tests**
  - Unit tests must cover scaling/ablation behavior, targeted indexing, and strict-mode failure paths.

### Go / No-Go

- Do not interpret intervention results if hook efficacy is not demonstrated on a small sanity batch.
- Do not continue if a targeted hook path silently falls back to module-level scaling in a "strict" validity run.

## Tokenization and Prompt-Resolution Diagnostics

Tokenization remains a major confounder for arithmetic experiments, especially when scoring final numeric outputs or per-digit targets.

### Required metrics

- average tokens per prompt
- fraction of prompts/operands using multi-token number encodings
- explicit target tokenization success rate
- single-token target filter rate (for control tasks that require explicit target-token scoring)
- fallback/context-mismatch counts in prompt-record resolution

### Go / No-Go

- Do not compare target-token causal metrics across prompt sets if tokenization filters differ materially.
- If explicit targets are required and filter rate is poor, redesign the prompt family before running large sweeps.

## Arithmetic Failure Taxonomy Metrics

These metrics are required for the operator-bottleneck program and should be logged per operator bucket (not only aggregated across all arithmetic prompts).

### Core metrics

- **Final-answer accuracy**
  - `accuracy_all`
  - parsed-subset accuracy (if parsing is used)
  - `parse_rate`
- **Per-digit correctness**
  - per-position exact-match rate
  - first-error-position distribution
- **Carry / borrow behavior**
  - carry error rate (addition/multiplication partial sums)
  - borrow error rate (subtraction)
  - cascading carry/borrow breakdown
- **Operator confusion / wrong-mode outputs**
  - incorrect operator pattern outputs (e.g., addition-like outputs on subtraction prompts)
- **Representation sensitivity**
  - metric deltas across formatting/serialization variants (spaced digits, tags, etc.)

### Go / No-Go

- Do not claim a "math improvement" if the change is explained primarily by parse-rate changes.
- Do not aggregate across operator buckets without also reporting per-bucket breakdowns.

## Causal Localization Metrics (Arithmetic-Specific)

These metrics are for arithmetic-specific ranking/localization (attention and MLP), replacing induction proxies as the primary selector for math interventions.

### Required metrics

- **Answer-token causal metrics**
  - answer-token logit delta (ablation / amplification / patching)
  - answer-token probability delta
  - next-token KL divergence (baseline vs intervention)
  - logit L1/L2 deltas
- **Per-digit target metrics** (when available)
  - per-digit target logit delta
  - per-digit target probability delta
  - digit-position-specific intervention effect sizes
- **Behavioral effect metrics**
  - accuracy delta vs baseline
  - delta vs matched-random component sets
  - operator-specific effect size deltas (target operator vs non-target operators)

### Necessity vs sufficiency criteria

For any localized component set, report both:
- **Necessity**: ablation harm (accuracy/logit/behavioral degradation)
- **Sufficiency**: amplification/patching improvement (or lack thereof)

Interpretation rule:
- Strong ablation harm plus weak amplification gain is a valid and important result (not a failed experiment).

## Control-Task Detector / Localizer Validation (General)

Plan A validated an induction-head detector/localizer pipeline using positive/negative control suites. Future arithmetic-localizer pipelines should follow the same pattern.

### Required control checks (adapt for the target component family)

- positive controls with known expected behavior
- matched negative controls with similar surface statistics but without the target structure
- matched-random component-set baselines
- non-silent effect checks (avoid all-zero causal metrics)
- separability checks (positive vs negative controls with CIs)

### Go / No-Go

- Do not interpret arithmetic interventions if control separability is absent.
- Do not use proxy-only rankings when causal metrics are zero/non-informative.

## Stability and Robustness

This section standardizes terminology and avoids overclaiming from prompt-order-only checks.

### Required terminology

- **Same-set shuffle invariance**
  - Rankings/effects are unchanged when the same prompt set is reordered.
  - This is useful, but it is not the same as resampling robustness.
- **Subsample stability**
  - Rankings/effects persist across different prompt subsets drawn from the same pool/family.
- **Family-heldout stability**
  - Rankings/effects learned on some prompt families generalize to held-out families.
- **Seed robustness**
  - Results remain directionally consistent across random seeds.

### Minimum robustness expectations (publication-facing)

- Report same-set shuffle invariance only with that exact label.
- Add at least one true subsample stability metric for any ranking used in arithmetic claims.
- Prefer family-heldout validation for localized operator heuristics.

## Reporting and Artifact Requirements

Every diagnostic or validity run should emit machine-readable outputs plus enough metadata to reproduce the result.

### Required metadata

- model name + revision (or local path + commit hash)
- tokenizer name/version
- dataset hash / prompt suite identity
- metric configuration (target token policy, epsilon, batch size, etc.)
- seed(s)
- git commit (for code provenance)

### Required outputs (examples)

- phase/gate summary JSON (single source of truth for pass/fail)
- detailed per-run JSONs (detector/localizer/control sweeps)
- manifests and logs
- optional human-readable summary/report markdown

### Interpretation hygiene

- Separate facts (artifact-backed metrics) from hypotheses (mechanistic interpretations).
- State whether a result supports a control-task validity claim, an arithmetic improvement claim, or a hypothesis-generation step.

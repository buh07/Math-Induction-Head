# LLM Math Failure as Heuristic Bottleneck and Control/Gating Limitation - Preliminary Report

## Abstract (Draft)

This report investigates why LLMs remain weak at arithmetic and mathematical reasoning through a mechanistic interpretability lens. We first present **Phase 1 (Steering Baseline / Validated Baseline)**, specifically the completed Phase 1 / Plan A validity tranche, which establishes a causally active induction-head targeting/steering pipeline on induction-style control tasks. We then show that, despite this validity result, induction-head amplification yields mostly null/mixed arithmetic gains in a minimal arithmetic sanity rerun, while ablation of validated top heads can be strongly harmful. These findings motivate **Phase 2 (Operator Heuristic Bottleneck Mainline)**, in which arithmetic failure is treated as an operator-specific heuristic bottleneck plus control/gating/composition limitation. We outline the next experimental program focused on arithmetic-specific causal localization, necessity/sufficiency testing, and CoT gating/composition analysis.

## Introduction

### Motivation

- LLMs can appear competent on narrow arithmetic slices but fail under composition, distribution shift, and harder reasoning tasks.
- CoT often improves performance, but the mechanism of that improvement is still unclear.
- Mechanistic evidence suggests multiple circuit families may contribute (induction-like scaffolding, arithmetic-specific heuristics, control/gating circuits).

### Updated research question

- Where is the arithmetic bottleneck in LLMs?
- Are induction heads the primary lever, or are operator-specific heuristic circuits plus control/gating failures the dominant explanation?

### Project status

- **Phase 1 (Steering Baseline / Validated Baseline)** is complete and now serves as a validated baseline/control track (Phase 1 / Plan A).
- **Phase 2 (Operator Heuristic Bottleneck Mainline)** is the active project direction.

## Methods

### Plan A: Induction-head validity tranche (completed baseline track)

- Model: `meta-llama/Meta-Llama-3-8B`
- Staged gates:
  - Phase 0 hook efficacy
  - Phase 1 control prompt filtering
  - Phase 2 detector validity (positive/negative controls + GSM plain/CoT comparison)
  - Phase 3 control-task steering validity
  - Phase 4 arithmetic sanity rerun (conditional on Phase 3 pass)
- Key artifacts:
  - `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/gate_summary.json`
  - `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase2_detector/phase2_summary.json`
  - `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase3_gate_summary.json`
  - `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase4_arithmetic_sanity.json`

### Plan B+ (next tranche): Operator heuristics / arithmetic bottleneck program

Planned methods (not yet executed in this report draft):
- operator-bucket dataset buildout (carry/borrow/subpattern taxonomy)
- arithmetic-specific causal localization (attention + MLP)
- necessity/sufficiency interventions with matched controls
- cross-operator specificity matrix
- CoT vs direct-answer circuit recruitment / gating comparisons

## Results

## Plan A Validity Results

### Gate outcomes (completed)

- Phase 0/1/2/3 gates passed
- `ready_for_multimodel_next_tranche: true`
- detector outputs show non-silent causal effects and positive-vs-negative separability
- control steering validity passes via amplification sensitivity and specificity criteria

### Key quantitative highlights (to tabulate)

- `rank_stability_spearman_top50 = 1.0` (with same-set shuffle invariance caveat)
- `effect_nonzero_rate_max = 0.3333`
- `gsm_plain_vs_cot_rank_stability ~= 0.964`
- strongest Phase 3 Top-vs-random amplification effect `~ +0.095` copy-target-prob delta (CI > 0)

## Arithmetic Sanity Outcomes (Null/Mixed Amplification, Strong Ablation Harm)

### Baseline arithmetic (Phase 4)

- `tier1_in_distribution = 0.77`
- `tier2_near_ood = 0.3667`

### Intervention pattern (Phase 4)

- top validated-head amplification produces small/non-significant changes in arithmetic accuracy
- top validated-head ablations can produce large negative deltas
- interpretation: validated induction-like heads matter for behavior but are not a robust arithmetic-improvement lever under simple amplification in this setup

## Pivot Rationale and Next Experiments

### Why the pivot is justified

- Control-task validity is established, so the weak arithmetic improvement signal is now interpretable.
- The project can make a stronger claim by shifting to operator-specific heuristic bottlenecks and CoT gating/composition rather than repeating induction amplification sweeps.

### Next experiments (planned)

- operator-bucket arithmetic failure taxonomy
- arithmetic-specific causal localization (attention + MLP)
- necessity/sufficiency + cross-operator specificity
- CoT vs direct-answer circuit recruitment/gating

## Discussion

### Refined hypothesis (current)

Induction-like heads appear to function as part of a reusable scaffolding/control system, but arithmetic failure is more plausibly explained by operator-specific heuristic bottlenecks and control/gating limitations. CoT likely helps by externalizing intermediate state and improving heuristic composition over multiple steps.

### Limitations (current draft)

- Plan A arithmetic analysis is a minimal sanity rerun, not a full arithmetic-localizer program.
- The Phase 2 rank-stability metric in Plan A is best interpreted as same-set shuffle invariance, not full resampling robustness.
- Cross-model replication of the pivoted operator-bottleneck program is pending.

### Claim hygiene

This draft should continue to separate:
- control-task validity claims
- arithmetic improvement claims
- correlational proxy shifts vs causal intervention evidence

## Appendix

### Reproducibility / runtime notes

- Long runs should be launched with tmux or a scheduler and write explicit status sidecars (`EXIT_CODE=`).
- Use generated manifests and phase/gate summaries as the source of truth.

### Planned figures/tables (placeholders)

- Plan A phase/gate summary table
- Phase 3 control steering effect-size table
- Phase 4 arithmetic sanity summary (baseline vs top/random/bottom sets)
- Operator-specific localization matrix (future Plan B+)
- CoT vs direct-answer circuit recruitment comparison (future Plan B+)

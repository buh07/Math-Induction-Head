# Research Overview: Arithmetic Failure as Heuristic Bottleneck + Control/Gating Problem

## Executive Summary

This project began as an induction-head steering program for arithmetic: if Chain-of-Thought (CoT) reasoning relies on induction-like circuitry, perhaps forcing or amplifying induction heads during direct arithmetic would improve math performance.

That is no longer the mainline hypothesis.

The repository is now split into:
- **Phase 1 (Steering Baseline / Validated Baseline)** - induction-head steering/reruns plus the completed Phase 1 / Plan A validity tranche
- **Phase 2 (Operator Heuristic Bottleneck Mainline)** - the post-pivot operator-specific heuristic bottleneck + CoT gating/composition program

Phase 1 / Plan A (the induction-head validity tranche) is complete and changes the scientific status of the project:
- the induction-head detector and steering pipeline are now causally validated on induction-style control tasks
- arithmetic amplification gains are mostly null/mixed in the tested steering family
- top validated induction-head ablations can be strongly harmful for arithmetic

This supports a pivot in the research program:
- **Induction heads remain important** as a validated scaffolding/control axis
- the primary explanation for LLM math failure is now treated as an **operator-specific heuristic bottleneck plus control/gating/composition limitation**, consistent with the bag-of-heuristics framing

Phase 1 / Plan A evidence anchors:
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/gate_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase2_detector/phase2_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase3_gate_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase4_arithmetic_sanity.json`

## Part 1: The Empirical Problem - Why LLM Math Still Fails

LLMs remain weak at arithmetic and mathematical reasoning despite strong performance on many language tasks. The important point for this project is not just that models fail, but that they fail in a patterned way:
- they can perform well on narrow in-distribution arithmetic slices
- they degrade sharply under modest distribution shifts (operand range, format, composition)
- they often improve under CoT prompting, but not in a way that implies robust algorithmic generalization

This motivates a mechanistic question rather than a benchmark-only question:
- what internal computations are supporting apparent math competence, and where do they fail?

The current working answer is no longer "insufficient induction-head activation" in general.

## Part 2: Bag of Heuristics as the Current Best Bottleneck Model

The strongest current framing for the arithmetic bottleneck is the "bag of heuristics" view (Nikankin et al., ICLR 2025): language models solve arithmetic using collections of prompt-conditional heuristics rather than learning a unified, algorithmic arithmetic procedure.

Why this framing is a good fit for this repository:
- it explains strong local performance with poor extrapolation
- it predicts operator-specific and subpattern-specific failure modes (carry, borrow, operand range)
- it suggests the bottleneck may live in arithmetic-specific component sets (especially MLP neurons and their interaction with attention), not just one attention head class

For this project, the practical consequence is:
- we should localize and intervene on **operator-specific heuristic circuitry** (addition/subtraction/multiplication and subpatterns) instead of treating induction heads as the default target for arithmetic improvement.

## Part 3: Where Induction Heads Fit (and Why They Are Not the Whole Story)

Induction heads still matter in this project, but their role is now more precise.

Current interpretation:
- induction heads are likely part of a **reasoning scaffold** (pattern continuation, retrieval/copying, step formatting, local sequence regularities)
- this overlaps with in-context learning (ICL) and likely parts of CoT token generation
- arithmetic correctness appears to require additional mechanisms beyond simply increasing induction-head activity

This is consistent with the Plan A pattern:
- validated top induction-like heads are causal on induction control tasks
- ablation of those validated heads can harm arithmetic (so they are involved)
- amplification of those same heads does not robustly improve arithmetic in the tested setup

This is a necessity-vs-sufficiency distinction, not a contradiction.

## Part 4: What Phase 1 / Plan A Established (Causal Validation Tranche)

Phase 1 / Plan A was designed as a validity-first tranche before making further mechanistic claims.

### What Plan A tested

- strict head-targeted hook correctness (no silent no-op behavior)
- induction-head detector validity using positive/negative control prompt suites
- control-task steering validity (Top-K detector-selected heads vs matched-random and other controls)
- minimal arithmetic sanity rerun using validated head sets

### What Plan A established (artifact-backed)

From `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/gate_summary.json`:
- Phase 0 hook efficacy gate: pass
- Phase 1 prompt-filter gate: pass
- Phase 2 detector-validity gate: pass
- Phase 3 steering-validity gate: pass
- `ready_for_multimodel_next_tranche: true`

From `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase2_detector/phase2_summary.json`:
- `rank_stability_spearman_top50 = 1.0`
- positive-vs-negative separability passes (CI lower bound > 0)
- `effect_nonzero_rate_max = 0.3333`
- `gsm_plain_vs_cot_rank_stability ~= 0.964`

From `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase3_gate_summary.json`:
- overall Phase 3 pass
- amplification sensitivity and specificity criteria pass repeatedly
- ablation-sensitivity sub-criterion (>=10 pp copy-accuracy drop vs random) does not pass in the tested ablation settings

From `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase4_arithmetic_sanity.json`:
- baseline `tier1_in_distribution = 0.77`
- baseline `tier2_near_ood = 0.3667`
- top-head amplification gains are small / non-significant
- top-head ablations can be strongly harmful (large negative deltas)

### What changed and why the project pivoted

Before Plan A, the main uncertainty was whether induction-head steering failures reflected invalid targeting, broken hooks, or truly weak arithmetic leverage.

After Plan A:
- control-task validity is established
- arithmetic gains remain weak under the tested induction-head amplification family

That is the reason for the pivot. The project is not abandoning induction heads; it is narrowing their role and moving the main bottleneck hypothesis to operator-specific heuristics plus control/gating/composition.

## Part 5: Updated Mechanistic Hypothesis for Math Failure

Updated hypothesis (mainline):
- LLM arithmetic failure is best explained by a combination of:
  1. **operator-specific heuristic bottlenecks** (addition, subtraction, multiplication subcircuits)
  2. **gating/composition failures** (selecting and chaining the right heuristics across steps)
  3. **representation/tokenization constraints** (number format and digit composition effects)

In this framing, induction heads are expected to be:
- important for scaffolding and retrieval-like behavior
- potentially necessary in some arithmetic contexts
- insufficient as a standalone improvement lever for arithmetic accuracy

This framing directly supports the next experiment family:
- localize arithmetic-relevant heads/neurons by operator and subpattern
- test necessity/sufficiency and cross-operator specificity

## Part 6: CoT as Externalized Compute + Heuristic Gating/Composition

The project now treats CoT benefits as a mechanistic phenomenon that likely depends on multiple subsystems.

Current working interpretation:
- CoT provides extra serial inference-time compute (more tokens, more steps)
- CoT externalizes intermediate state into the context window, allowing re-reading through attention
- CoT improves gating/composition over a bag of heuristics (including step formatting and retrieval-like continuation)
- CoT may recruit induction-like scaffolding, but arithmetic correctness also depends on arithmetic-specific compute/control circuitry

This makes a concrete prediction for future experiments:
- direct-answer vs CoT should differ in circuit recruitment patterns, and those differences should not reduce to induction-head activation alone

## Part 7: New Experimental Program - Operator Heuristic Bottleneck Mapping

This section defines **Phase 2 (Operator Heuristic Bottleneck Mainline)**.

This is the new mainline program.

### A. Build operator-bucket datasets and failure taxonomy

Create datasets with explicit buckets and annotations:
- Addition: no-carry / single-carry / cascading-carry / length extrapolation
- Subtraction: no-borrow / borrow / cascading-borrow / negatives
- Multiplication: table lookup / partial-product / carry in partial sums / multi-digit composition

Track metrics beyond final accuracy:
- per-digit correctness
- carry/borrow error rate
- operator confusion
- parse-rate vs correctness separation

### B. Arithmetic-specific causal localization (attention + MLP)

Localize components directly against arithmetic targets:
- answer-token logit effects
- per-digit target logit effects
- patching/ablation effect sizes
- layer-window and token-position specificity (especially late layers / late tokens)

This program should treat attention heads and MLP neurons as first-class candidates, with MLP neurons likely central under the bag-of-heuristics framing.

### C. Necessity vs sufficiency interventions

For each operator-specific candidate set:
- ablation (necessity)
- amplification or patching (sufficiency)
- rescue tests when practical

Crucial requirement:
- measure cross-operator specificity (e.g., addition-set impacts addition more than subtraction/multiplication)

### D. CoT gating/composition experiments

Compare direct-answer and CoT prompts on matched arithmetic instances:
- circuit recruitment differences
- component sensitivity differences
- whether CoT benefits are explained by better heuristic gating/composition vs stronger induction-like scaffolding alone

## Part 8: Decision Gates, Failure Modes, and Interpretation Rules

### Gate philosophy (carried forward from Plan A)

Every mechanistic claim should pass staged gates:
- instrumentation works (no silent no-ops)
- detector/localizer has causal validity (not proxy-only)
- control specificity beats random/matched controls
- arithmetic improvement claims are made only after the targeting/localization gates pass

### Interpretation rules (important)

- Do not overclaim from null amplification results.
  - Allowed: induction-head amplification did not robustly improve arithmetic in the tested setup.
  - Not allowed: induction heads do not matter for math.

- Distinguish necessity from sufficiency.
  - Strong ablation harm plus weak amplification gain is a meaningful pattern.

- Distinguish same-set shuffle invariance from full robustness.
  - The current `rank_stability_spearman_top50 = 1.0` is compatible with reordering the same prompt pool, not necessarily resampling robustness.
  - Relevant implementation paths: `scripts/phase1/run_head_validity_suite.py:952`, `scripts/phase1/run_head_validity_suite.py:992`, `src/induction_detection.py:240`, `src/induction_detection.py:878`.

### Failure modes to plan for in the pivoted program

- operator-localized components are unstable across prompt families
- ablation shows necessity but amplification remains weak (gating bottleneck, not component strength)
- operator specificity is poor (candidate sets are generic syntax/format components)
- CoT improvements are mostly formatting and parseability, not arithmetic compute

These are all informative outcomes and should be documented as such.

## Part 9: Multi-Model Replication Path

Multi-model replication remains part of the project plan, but the order matters.

Current replication principle:
- validate the mainline targeting/localization method in one strong model first (Llama-3-8B)
- then replicate on one additional model (Gemma-2B recommended next)
- only then expand to weaker contrast models (e.g., GPT-2 / Pythia)

Plan A already established a reusable baseline/control tranche and marked readiness for the next tranche (`ready_for_multimodel_next_tranche: true`), but the **new** operator-specific program should replicate the same validity discipline before broad multi-model claims.

## Appendix: Historical Induction-Forcing Hypothesis (Archived Link)

The original induction-first overview (preserved for provenance and comparison) is archived at:
- `docs/archive/overview_induction_head_hypothesis_20260225.md`

That document reflects the earlier framing in which inducing induction-head activation was the primary proposed arithmetic intervention. The current `overview.md` supersedes it as the mainline theory/program document.

## Literature Anchors for This Overview

- Nikankin et al., *Arithmetic Without Algorithms: Language Models Solve Math with a Bag of Heuristics* (ICLR 2025): https://arxiv.org/abs/2410.21272
- Olsson et al., *In-context Learning and Induction Heads* (2022): https://arxiv.org/abs/2209.11895
- Cabannes et al., *Iteration Head: A Mechanistic Study of Chain-of-Thought* (2024): https://arxiv.org/abs/2406.02128
- Stolfo et al., *A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis* (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.435/
- Mamidanna et al., *All for One and One for All* (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.1565/
- Optional follow-up: Zhang et al., *Fine-Grained Manipulation of Arithmetic Neurons in Language Models* (BlackboxNLP 2025): https://aclanthology.org/2025.blackboxnlp-1.27/

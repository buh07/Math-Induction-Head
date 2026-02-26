# Multi-Model Replication Plan (Operator Heuristic Bottleneck)

This document defines the replication strategy after the post-Plan-A pivot.

Repository phase context:
- **Phase 1 (Steering Baseline / Validated Baseline)** = completed induction-head steering baseline + Phase 1 / Plan A validity tranche
- **Phase 2 (Operator Heuristic Bottleneck Mainline)** = current mainline replication target

The project now treats the completed induction-head validity tranche (Plan A) as a **validated baseline / comparison axis**. The main replication target is the new **operator-specific heuristic bottleneck** program.

## Replication Principles

- Replicate validity discipline before replicating large intervention sweeps.
- Do not assume a targeting/localization method transfers across models without re-validating it.
- Separate replication of control-task validity from replication of arithmetic-improvement claims.
- Use the same terminology across models:
  - validated baseline
  - operator-specific heuristic bottlenecks
  - necessity vs sufficiency
  - same-set shuffle invariance (when applicable)
  - CoT gating/composition

## Model Prioritization Criteria

Prioritize models by:

- **Arithmetic baseline strength**
  - enough signal to distinguish bucketed arithmetic behaviors
- **Tokenization characteristics**
  - can expose useful contrasts in number tokenization behavior
- **Runtime feasibility**
  - practical turnaround for localization + intervention sweeps
- **Compatibility with current hook stack**
  - stable attention/MLP hook integration and test coverage

## Recommended Replication Order

1. **`meta-llama/Meta-Llama-3-8B`** (already validated baseline)
   - Use as the reference model for developing the operator-bottleneck pipeline.
2. **`google/gemma-2b`** (next recommended replication)
   - Strong enough arithmetic baseline in prior runs to provide a useful contrast.
3. **One weaker contrast model** (e.g., `gpt2` or `EleutherAI/pythia-1.4b`)
   - Useful for boundary conditions and negative/low-capability comparisons.

Optional later additions:
- architecture variants with different tokenization/attention layouts
- larger models only after the operator-localizer interface is stable

## Replication Gates (New Mainline Program)

Before running full arithmetic intervention sweeps on a new model, require these gates:

### Gate 1: Instrumentation validity

- Hook paths work for the model architecture (attention + MLP targeting)
- Strict no-op checks pass
- Sanity causal effects are non-zero on small control batches

### Gate 2: Localizer validity (arithmetic-specific)

- Arithmetic-specific causal metrics are non-zero and reproducible
- Localized component ranking beats matched-random on at least one operator bucket
- Robustness includes true subsample/family-heldout checks (not only same-set shuffle invariance)

### Gate 3: Control specificity / negative controls

- Positive-control effects exceed matched negative controls where applicable
- Arithmetic-localized sets show operator specificity or a clearly interpretable failure mode

### Gate 4: Only then run full arithmetic interventions

- Necessity/sufficiency sweeps on operator-localized component sets
- Cross-operator specificity matrix
- CoT vs direct-answer comparisons (selected subsets)

## What Counts as Replication Success

Replication success is not just "accuracy improves."

### A. Localization stability success

- Rankings/effects are directionally stable across seeds and subsamples
- Held-out family tests do not collapse immediately
- Metric outputs are non-silent and artifact-backed

### B. Operator specificity success

- At least one localized set shows stronger impact on its target operator than on non-target operators
- Results beat matched-random controls with reported CIs

### C. Intervention directionality success

- Necessity and/or sufficiency pattern is consistent with the hypothesized component role
- Null amplification with strong ablation is still an informative replicated pattern (necessity without sufficiency)

## Suggested Replication Sequence (Practical)

### Stage 1 - Baseline validation on the new model

- Run a reduced validity tranche for the arithmetic-localizer pipeline (not full Plan A induction-only workflow).
- Confirm hook correctness, metric sanity, and control separability.

### Stage 2 - Operator-localizer runs

- Run localization on operator buckets (addition/subtraction/multiplication subsets).
- Save candidate sets and robustness summaries.

### Stage 3 - Necessity/sufficiency sweeps

- Run ablation/amplification/patching on localized sets.
- Include matched-random and induction-head baseline comparison sets.

### Stage 4 - Selected CoT gating/composition comparisons

- Run matched direct-answer vs CoT subsets to test recruitment differences.
- Keep this targeted until earlier stages stabilize.

## Artifact / Logging Requirements for Replication

Every model replication run should record:
- model name and revision (or local path)
- tokenizer version
- GPU allocation and runtime command
- dataset hashes / prompt suite identifiers
- metric configuration and seeds
- git commit hash

Required outputs:
- run manifest
- phase/gate summary JSON(s)
- detailed localization/intervention outputs
- logs and status sidecars for long tmux/scheduler runs

## Historical Note (Legacy / Comparison Context)

The older GPT-2 induction amplification quickstart remains useful as historical context and a comparative control track, but it is no longer the primary replication path for the project. The mainline replication target is the operator-heuristic bottleneck program described above.

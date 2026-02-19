# Diagnostic Metrics & Gates

This document defines the quantitative checks required before running costly
experiments. All diagnostics are implemented with the Week 2 infrastructure.

## Attention & Neuron Hooks

- **Configuration**: `AttentionHookConfig` and `NeuronHookConfig` specify
  `(layer, index, scale)` tuples. Hook sets are applied via `HookManager`.
- **Go / No-Go**:
  - Hooks must be unit-tested with synthetic tensors.
  - Any runtime hook set must include explicit provenance (who generated it,
    which dataset drove the selection).

## Staged Ablation Runner

- **Stages**: An `AblationStage` lists layers plus a baseline name (`mean`,
  `zero`, `identity` provided). The `StagedAblationRunner` materializes
  replacements per layer.
- **Metrics**:
  - Baseline output magnitude (mean of replacements).
  - Layer coverage (all targeted layers present in activations map).
  - Comparison between baselines (`zero` drop vs `mean` drop).
- **Go / No-Go**:
  - Abort if any requested layer is missing.
  - Require at least two baseline strategies to be tested before promoting a stage.

## Tokenization Diagnostics

- **Analyzer**: `tokenization_diagnostics.analyze_prompts` tokenizes entire
  prompts plus extracted numbers.
- **Metrics**:
  - Average tokens per prompt.
  - Fraction of operands that decompose into multiple tokens.
  - Total operand count.
- **Go / No-Go**:
  - Proceed with multi-token experiments only after multi-token share exceeds 10%.
  - If single-token share >90%, expand dataset or adjust tokenizer before running
    multi-step arithmetic trials.

## Reporting

- Include diagnostic JSON dumps alongside experiment manifests.
- Record tokenizer/version, dataset hash, and git commit in every diagnostic run.

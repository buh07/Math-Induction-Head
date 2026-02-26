# Math Induction Head (Pivoted Research Program)

This repository is now organized into two project phases:

- **Phase 1 (Steering Baseline / Validated Baseline)**: induction-head steering/reruns plus the completed Phase 1 / Plan A validity tranche
- **Phase 2 (Operator Heuristic Bottleneck Mainline)**: the post-pivot operator-specific heuristic bottleneck + CoT gating/composition program

The project started as an induction-head steering program for arithmetic. After the completed Phase 1 / Plan A validity tranche, induction-head targeting and steering are treated as a **validated baseline** and **comparison axis**, while the mainline direction pivots to **operator-specific heuristic bottlenecks** and **CoT gating/composition**.

This README is the primary project entrypoint. It covers the scientific framing, what is already established in Phase 1, how to reproduce/read the validated tranche outputs, and how to run Phase 2 work as it is implemented.

## Current Status (What Is Established)

- **Phase 1 (Steering Baseline / Validated Baseline)** is complete enough to serve as a validated comparison axis; Phase 1 / Plan A completed successfully with `EXIT_CODE=0` (`logs/phase1/canonical/20260225_120553_head_validity_planA_gpu01.status`).
- The induction-head targeting/steering pipeline is causally validated on control tasks (hook efficacy, detector validity, control steering validity all passed) (`results/phase1/canonical/head_validity_run_20260225_120553_gpu01/gate_summary.json`).
- Arithmetic amplification gains in the Phase 4 sanity rerun are mostly null/mixed, while top-head ablations are strongly harmful (`results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase4_arithmetic_sanity.json`).
- The project is now pivoting to **operator-specific heuristic bottlenecks** and **CoT gating/composition** as the primary research direction.
- Phase 1 artifacts are organized by trust status under `results/phase1/`: `canonical`, `provisional_pre_fix`, and `failed_or_partial`.
- The Plan A run explicitly marked readiness for the next tranche (`ready_for_multimodel_next_tranche: true`) (`results/phase1/canonical/head_validity_run_20260225_120553_gpu01/gate_summary.json`).

Primary artifact anchors (repo root = `Math Induction Head/`):
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/gate_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase2_detector/phase2_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase3_gate_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase4_arithmetic_sanity.json`

## Phase Reorganization / Migration Notes

This repository was reorganized to make the phase split explicit.

- **Phase 1 (Steering Baseline / Validated Baseline)** assets now live under phase-labeled paths such as:
  - `results/phase1/...`
  - `logs/phase1/...`
  - `scripts/phase1/...`
  - `configs/phase1/...`
  - `prompts/phase1/...`
- **Phase 2 (Operator Heuristic Bottleneck Mainline)** placeholders and future assets live under:
  - `results/phase2/...`
  - `logs/phase2/...`
  - `scripts/phase2/...`
  - `configs/phase2/...`
  - `prompts/phase2/...`

Migration examples:
- `results/head_validity_run_20260225_120553_gpu01/...` -> `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/...`
- `logs/20260225_120553_head_validity_planA_gpu01.log` -> `logs/phase1/canonical/20260225_120553_head_validity_planA_gpu01.log`
- `scripts/run_head_validity_suite.py` -> `scripts/phase1/run_head_validity_suite.py` (root wrapper kept temporarily)

Old path references in older notes are superseded by:
- `results/phase1/migration_map.json`
- `logs/phase1/migration_map.json`

## Research Question (Updated)

Original narrow question:
- Can forcing induction-head activation improve arithmetic and math reasoning?

Updated broader question (mainline):
- Why are LLMs still bad at math, and where is the bottleneck?
- Are the main bottlenecks best explained by **operator-specific heuristic circuits** (addition/subtraction/multiplication heuristics and subpatterns) plus **control/gating/composition failures**, rather than insufficient induction-head activation?

Working interpretation after Phase 1 / Plan A:
- Induction-like heads matter as a causal scaffolding/routing signal.
- They are not, by themselves, a robust arithmetic-improvement lever under the tested steering family.

## Why the Project Pivoted

### Phase 1 (Steering Baseline / Validated Baseline)

The induction-head work remains important and is not discarded. It is now the validated baseline track because it is useful for:
- hook correctness and no-op detection
- causal targeting validation
- control-task sensitivity checks
- comparative controls for future arithmetic-specific interventions

### Phase 2 (Operator Heuristic Bottleneck Mainline)

The mainline direction pivots because Plan A showed a gap between control-task validity and arithmetic improvement:
- operator-specific heuristics (addition/subtraction/multiplication and subpatterns)
- attention + MLP localization on arithmetic tasks
- CoT gating/composition and externalized compute

This is a direct response to the Phase 1 / Plan A outcome: validated steering on induction controls did not translate into robust arithmetic gains.

## Core Hypothesis Going Forward

The current hypothesis is:
- LLM math failures are primarily driven by **operator-specific heuristic bottlenecks** and **control/gating limitations**, not by insufficient activation of induction heads alone.

Sub-hypotheses:
- Induction heads provide reusable scaffolding (pattern continuation, copying, step formatting).
- Arithmetic correctness depends on additional arithmetic-specific components (likely including MLP neurons and late-layer circuitry).
- CoT helps by increasing serial compute, externalizing intermediate state, and improving heuristic gating/composition.

## What We Learned from Phase 1 / Plan A (Induction-Head Validity Tranche)

### High-level conclusion

Plan A succeeded at the validity objective:
- It established that the induction-head detector and steering pipeline are causally active on induction-style control tasks.
- It did **not** establish robust arithmetic gains from induction-head amplification.

### Concrete numerical summary (artifact-backed)

Phase 2 detector validity (`results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase2_detector/phase2_summary.json`):
- `rank_stability_spearman_top50 = 1.0`
- separability passed (positive vs negative control median delta CI lower bound > 0)
- `effect_nonzero_rate_max = 0.3333`
- `gsm_plain_vs_cot_rank_stability ~= 0.964`

Phase 3 control steering validity (`results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase3_gate_summary.json`):
- overall gate: `passes = true`
- amplification sensitivity: passes in all 12 applicable checks
- specificity (positive controls > negative controls): passes in all 12 applicable checks
- ablation sensitivity sub-criterion (>=10 pp copy-accuracy drop vs random): failed in all 4 ablation checks
- strongest amplification effect (Top-K vs random): `K=10`, `scale=2.0`, `downscale_others=0.9`, mean copy-target-prob delta `~ +0.095` (CI > 0)

Phase 4 arithmetic sanity (`results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase4_arithmetic_sanity.json`):
- baseline `tier1_in_distribution = 0.77`
- baseline `tier2_near_ood = 0.3667`
- top-head amplification gains are small and non-significant (e.g., `K=10`, `scale=1.5`: `tier1=0.78` / `+0.01`, `tier2=0.3833` / `+0.0167`, both CIs cross 0)
- top-head ablations are strongly harmful (e.g., `tier1` top ablations drop to `0.30` and `0.32`; `tier2` `K=10` top ablation drops to `0.15`)

### What changed scientifically

This tranche changed the status of the induction-head idea from:
- "promising but not yet trustworthy"

to:
- "causally validated on induction controls, but not a robust arithmetic-improvement lever in the tested intervention family"

## How This Connects to the Bag-of-Heuristics Framework

This repository now uses the bag-of-heuristics framing as the main bottleneck model for arithmetic failure.

Practical implication for this project:
- stop treating induction-head amplification as the default fix for math
- start localizing and intervening on **operator-specific heuristic circuitry** (addition/subtraction/multiplication and subpatterns such as carry/borrow)

Mechanistic interpretation after Phase 1 / Plan A:
- induction heads overlap with reasoning/ICL-style scaffolding
- arithmetic performance likely depends on additional components (attention + MLP + late-layer control) that are not improved by simple head amplification

## Why Chain-of-Thought Helps Math (Current Mechanistic Interpretation)

Current working explanation (to be tested directly):
- CoT gives the model more serial inference-time compute.
- CoT externalizes intermediate state into tokens that can be re-read through attention.
- CoT improves gating/composition over a bag of heuristics (including, but not limited to, induction-like scaffolding).
- CoT may improve step formatting and continuation via induction-like heads, while arithmetic correctness still depends on arithmetic-specific compute/control circuits.

This is why the project now distinguishes:
- induction/scaffolding validity (already established in Plan A)
- arithmetic improvement mechanisms (new mainline direction)

## Current Experimental Strategy (Operator Heuristic Bottleneck Mapping)

The active program is now an operator-circuit program, not an induction-only program.

### 1) Operator-bucket datasets (failure anatomy)

Build/expand arithmetic datasets with explicit buckets:
- addition: no-carry, single-carry, cascading-carry, digit-length extrapolation
- subtraction: no-borrow, borrow, cascading-borrow, negatives
- multiplication: table lookup cases, partial-product carry, multi-digit composition
- later: division/modulo and mixed-operation composition

Required outputs per bucket:
- final-answer accuracy
- per-digit correctness
- carry/borrow error rate
- parse-rate separated from correctness

### 2) Arithmetic-specific causal localization

Localize components directly on arithmetic outputs instead of using induction proxies as the primary selector.

Component categories to test:
- attention heads
- MLP neurons
- late-layer subgraphs / layer windows
- token-position-specific effects (especially late tokens / answer token)

Metrics:
- answer-token logit delta
- per-digit target logit delta
- patch/ablation effect sizes
- operator-specific specificity (e.g., addition-set affects addition more than subtraction)

### 3) Necessity vs sufficiency tests

For each candidate component set:
- ablation (necessity)
- amplification/patching (sufficiency)
- rescue tests when possible (correct-run patch into incorrect-run trajectory)

### 4) Cross-task specificity matrix

Every intervention family should be evaluated as a matrix:
- rows = localized component sets (`add`, `sub`, `mul`, random matched, induction baseline set)
- columns = operator tasks/buckets
- cells = ablation delta, amplification delta, CI

### 5) CoT gating/composition experiments

Compare direct-answer vs CoT on the same arithmetic instances:
- measure circuit recruitment differences
- test whether CoT benefits are due to gating/composition rather than stronger induction-head activation alone

## Reproducing the Latest Validated Results

This section is for **Phase 1 (Steering Baseline / Validated Baseline)**, specifically Phase 1 / Plan A.

### Prerequisites

- Working directory: repo root (`/scratch2/f004ndc/Math Induction Head`)
- Python venv with project dependencies installed (`.venv`)
- Access to the target model (`meta-llama/Meta-Llama-3-8B`) in the configured cache path or via HF auth
- 2+ GPUs recommended for the Plan A runtime used here (the validated run used GPUs `0,1`)

### A. Read existing artifacts (no rerun)

Use this first if you only need to inspect the validated tranche outputs.

```bash
cd '/scratch2/f004ndc/Math Induction Head'
python - <<'PY'
import json
from pathlib import Path
root = Path('results/phase1/canonical/head_validity_run_20260225_120553_gpu01')
for rel in [
    'gate_summary.json',
    'phase2_detector/phase2_summary.json',
    'phase3_gate_summary.json',
    'phase4_arithmetic_sanity.json',
]:
    p = root / rel
    print(f'\n=== {rel} ===')
    print(json.loads(p.read_text()))
PY
```

Healthy/expected files:
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/gate_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase2_detector/phase2_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase3_gate_summary.json`
- `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/phase4_arithmetic_sanity.json`

### B. Rerun Plan A (`scripts/phase1/run_head_validity_suite.py`)

This reproduces the validated baseline-style tranche (induction-head control validation + arithmetic sanity).

```bash
cd '/scratch2/f004ndc/Math Induction Head'
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/phase1/canonical/${TS}_head_validity_planA.log"
STATUS="logs/phase1/canonical/${TS}_head_validity_planA.status"
OUT="results/phase1/canonical/head_validity_run_${TS}"

mkdir -p logs/phase1/canonical results/phase1/canonical

tmux new -d -s mih_planA_validity "bash -lc '
  cd /scratch2/f004ndc/Math\ Induction\ Head
  echo START \$(date -Is) | tee -a \"$LOG\"
  while true; do
    echo HEARTBEAT \$(date -Is) | tee -a \"$LOG\"
    sleep 300
  done &
  HB_PID=\$!
  trap \"kill \$HB_PID >/dev/null 2>&1 || true\" EXIT
  .venv/bin/python scripts/phase1/run_head_validity_suite.py \
    --model meta-llama/Meta-Llama-3-8B \
    --cache-dir ../LLM\ Second-Order\ Effects/models \
    --devices 0,1 \
    --batch-size 8 \
    --seed-list 0,1 \
    --output-root \"$OUT\" >> \"$LOG\" 2>&1
  CODE=\$?
  echo EXIT_CODE=\$CODE \$(date -Is) | tee -a \"$LOG\"
  echo EXIT_CODE=\$CODE > \"$STATUS\"
  exit \$CODE
'"
```

### Monitoring a live run

```bash
tmux attach -t mih_planA_validity
```

```bash
tail -f '/scratch2/f004ndc/Math Induction Head/logs/phase1/canonical/<timestamp>_head_validity_planA.log'
```

Healthy progress signals:
- `run_manifest.json` appears first
- `phase0_debug.json` and `phase1_prompt_suites.json` appear next
- `phase2_detector/*.json` files land during the long detector phase
- `phase2_detector/phase2_summary.json` appears before Phase 3 starts
- final success is `EXIT_CODE=0` plus `gate_summary.json`

### How to detect success vs failure

Success:
- `.status` sidecar contains `EXIT_CODE=0`
- `results/.../gate_summary.json` exists
- `results/.../phase2_detector/phase2_summary.json` exists
- if Phase 3 passes, `phase4_arithmetic_sanity.json` exists (expected in current validated run)

Failure:
- `.status` missing or nonzero `EXIT_CODE`
- `gate_summary.json` missing
- `phase2_summary.json` missing (run likely interrupted before detector completion)

## Running the Next Tranche (Operator-Circuit Pivot)

The operator-bottleneck tranche is **Phase 2 (Operator Heuristic Bottleneck Mainline)**. The orchestration script for the full pivot program is **not implemented yet**. Use this section as the operational contract for the upcoming implementation.

### Prerequisites (same operational baseline as Plan A)

- `.venv` with repo dependencies
- target model weights available (Llama-3-8B first)
- GPU access (2+ GPUs recommended for long causal sweeps)
- enough disk for JSON outputs and logs under `results/` and `logs/`

### Planned operator-bucket inputs (required before full run)

- addition buckets: no-carry / single-carry / cascading-carry
- subtraction buckets: no-borrow / borrow / cascading-borrow
- multiplication buckets: table / partial-product / carry in partial sums
- metadata per prompt:
  - operator label
  - bucket label
  - expected answer
  - (where relevant) per-digit targets and carry/borrow annotations

### Placeholder command template (not implemented yet)

```bash
cd '/scratch2/f004ndc/Math Induction Head'
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/phase2/${TS}_operator_bottleneck.log"
STATUS="logs/phase2/${TS}_operator_bottleneck.status"
OUT="results/phase2/operator_bottleneck_run_${TS}"

# Placeholder interface - expected future script, not present yet.
# Do not run until implemented.
mkdir -p logs/phase2 results/phase2
.venv/bin/python scripts/phase2/run_operator_bottleneck_suite.py \
  --model meta-llama/Meta-Llama-3-8B \
  --devices 0,1 \
  --dataset-config configs/phase2/operator_buckets_llama3.example.yaml \
  --output-root "$OUT"
```

### What the next tranche must measure (minimum)

- arithmetic-specific localization stability (including subsample and family-heldout checks)
- necessity vs sufficiency for localized operator-specific components
- cross-operator specificity matrix
- direct-answer vs CoT circuit recruitment differences (CoT gating/composition)

## Artifact Layout and How to Read Results

This section describes the current **Phase 1 (Steering Baseline / Validated Baseline)** artifact layout, using the canonical Phase 1 / Plan A run at `results/phase1/canonical/head_validity_run_20260225_120553_gpu01/`.

Trust-status buckets for Phase 1 artifacts:
- `results/phase1/canonical/`
- `results/phase1/provisional_pre_fix/`
- `results/phase1/failed_or_partial/`
- path migration index: `results/phase1/migration_map.json` (and `logs/phase1/migration_map.json`)

- `run_manifest.json`
  - Run metadata (model, seeds, output root, config-like runtime parameters).
  - Read this first to verify which run you are looking at.

- `phase0_debug.json`
  - Hook efficacy sanity check on small prompt batches.
  - Confirms head-targeted interventions produce non-zero causal metrics (e.g., KL / logit deltas).
  - If this fails, later phases should not be trusted.

- `phase1_prompt_suites.json`
  - Control prompt suite generation + tokenization filtering statistics.
  - Reports family counts and single-token target filter rates.
  - Also shows fallback tokenization counts (`dropped_context_mismatch_fallback`).

- `phase2_detector/`
  - Detector runs over positive/negative controls and GSM plain/CoT prompt sets.
  - Files like `positive_seed0.json`, `negative_seed1.json`, etc. are per-run detector outputs.
  - `phase2_summary.json` is the Phase 2 source of truth for detector validity, separability, and ranking metrics.

- `head_sets.json`
  - Derived head sets used in Phase 3/4 (`top`, `random_matched`, `high_match_low_causal`, `bottom`) for each `K`.
  - Includes provenance (`source`, seeds, schema version) for replication.

- `phase3_control_sweeps.json`
  - Full control-task steering sweep outputs across `K`, `scale`, and `downscale_others` settings.
  - Use this for detailed effect-size analysis beyond the gate summary.

- `phase3_gate_summary.json`
  - Phase 3 gate decision summary and criterion-level pass/fail results.
  - This is the source of truth for the control steering validity gate.

- `phase4_arithmetic_sanity.json`
  - Minimal arithmetic sanity rerun (after Phase 3 passes).
  - Includes baseline metrics plus condition results for `top`, `random_matched`, `bottom`, and `high_match_low_causal` sets.
  - Use this to assess whether validated induction targets change arithmetic performance.

- `gate_summary.json`
  - Final tranche summary across phases.
  - Single source of truth for whether the run passed the required gates and whether it is ready for the next tranche.

- `replication_protocol.md`
  - Run-generated handoff document for repeating the validated baseline track on future models.

## Metrics and Decision Gates

### Phase 1 / Plan A gate logic (current validated baseline track)

- Phase 0 hook efficacy gate
  - Requires non-trivial causal effects under head ablation on positive controls.
- Phase 1 prompt filter gate
  - Requires adequate single-token target retention for control prompts.
- Phase 2 detector validity gate
  - Requires non-silent causal effects, positive-vs-negative separability, and ranking stability.
- Phase 3 steering validity gate
  - Requires at least one successful steering criterion (amplification sensitivity and/or specificity), not necessarily all sub-criteria.
- Phase 4 arithmetic sanity
  - Runs only after Phase 3 passes; this phase is interpretive, not the validity gate itself.

### Mainline gate logic (operator-bottleneck direction; planned)

Future arithmetic-specific runs should adopt explicit gates for:
- localization validity (operator-specific causal signal > random)
- robustness (subsample + family-heldout stability, not just same-set shuffle invariance)
- necessity vs sufficiency separation
- cross-operator specificity
- CoT gating/composition evidence (direct vs CoT circuit recruitment differences)

## Known Caveats / Interpretation Rules

### 1) `rank_stability_spearman_top50 = 1.0` is plausible, but limited

This value is not automatically an error in the current run. It is plausible because the Phase 2 detector seeds reused the same cached control prompt pool and primarily changed ordering, not content.

Relevant code paths:
- control pool generation once in `scripts/phase1/run_head_validity_suite.py:952`
- detector runs over cached records in `scripts/phase1/run_head_validity_suite.py:992`
- record shuffle helper in `src/induction_detection.py:240`
- top-k rank stability computation in `src/induction_detection.py:878`

Interpretation rule:
- Treat the current metric as **same-set shuffle invariance**, not full resampling robustness.
- Next tranche must add true subsample and family-heldout stability checks before making stronger claims about ranking stability.

### 2) No overclaiming induction-head results

Allowed claim:
- induction-head amplification did not robustly improve arithmetic in the tested setup

Not allowed claim:
- induction heads do not matter for math

Reason:
- Phase 4 top-head ablations were strongly harmful, which indicates involvement/necessity in at least part of the arithmetic behavior.

### 3) Separate facts, hypotheses, and planned work

- Facts = artifact-backed numbers in this README and generated JSON outputs
- Hypotheses = mechanistic interpretations (e.g., operator heuristics + CoT gating/composition)
- Planned work = future operator-bottleneck tranche and multi-model replication

## Documentation Map

Primary docs (Phase 2-facing top-level docs):
- `README.md` (this file): current status, interpretation, and operations
- `overview.md`: detailed theory and experimental program (pivoted)
- `ROADMAP.md`: project-level execution roadmap (pivoted)
- `TODO.md`: active execution tracker + historical completed checklist

Supporting docs:
- `docs/phase1/README.md`: Phase 1 (Steering Baseline / Validated Baseline) index and trust-status pointers
- `docs/phase2/README.md`: Phase 2 (Operator Heuristic Bottleneck Mainline) index / placeholders
- `results/phase1/README.md`: Phase 1 result trust-status buckets + canonical anchors
- `logs/phase1/README.md`: Phase 1 log trust-status buckets + migration notes
- `docs/diagnostics.md`: diagnostics and gate metrics (including stability/robustness terminology)
- `docs/multi_model_plan.md`: replication strategy for the new operator-bottleneck program
- `reports/phase2/PUBLICATION_DRAFT.md`: preliminary paper/report scaffold
- `reports/PUBLICATION_DRAFT.md`: temporary pointer stub to the Phase 2 draft
- `docs/archive/overview_induction_head_hypothesis_20260225.md`: archived induction-first overview for provenance

## Literature Anchors

Primary references guiding the current pivot:
- Nikankin et al., *Arithmetic Without Algorithms: Language Models Solve Math with a Bag of Heuristics* (ICLR 2025): https://arxiv.org/abs/2410.21272
- Olsson et al., *In-context Learning and Induction Heads* (2022): https://arxiv.org/abs/2209.11895
- Cabannes et al., *Iteration Head: A Mechanistic Study of Chain-of-Thought* (2024): https://arxiv.org/abs/2406.02128
- Stolfo et al., *A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis* (EMNLP 2023): https://aclanthology.org/2023.emnlp-main.435/
- Mamidanna et al., *All for One and One for All: Understanding LLMs' Direct Mental Math through Conceptual Abstraction* (EMNLP 2025): https://aclanthology.org/2025.emnlp-main.1565/
- (Optional follow-up) Zhang et al., *Fine-Grained Manipulation of Arithmetic Neurons in Language Models* (BlackboxNLP 2025): https://aclanthology.org/2025.blackboxnlp-1.27/

These references support the current project split:
- induction heads as scaffolding/control baseline
- arithmetic-specific heuristics and gating/composition as the primary bottleneck hypothesis

## Contributing / Workflow Expectations

- Treat Phase 1 (Steering Baseline / Validated Baseline) induction-head results as the current validated baseline, not as the Phase 2 mainline intervention strategy.
- Use artifact-backed claims in docs and reports; cite concrete JSON/log paths.
- Separate control-task validity claims from arithmetic-improvement claims.
- Keep terminology consistent across docs:
  - `Phase 1 (Steering Baseline / Validated Baseline)`
  - `Phase 2 (Operator Heuristic Bottleneck Mainline)`
  - `validated baseline`
  - `operator-specific heuristic bottlenecks`
  - `necessity vs sufficiency`
  - `same-set shuffle invariance`
  - `CoT gating/composition`
- Follow `CONTRIBUTING.md` for environment setup, tests, and logging requirements.

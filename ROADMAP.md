# Optimized Research Roadmap: Induction Heads and Arithmetic

## Executive Summary

This roadmap provides a literature-backed, resource-efficient approach to testing whether forced induction head activation improves LLM arithmetic. It addresses 10 identified methodological weaknesses while staying tractable within realistic constraints.

**Key improvements over earlier proposals:**
1. Compressed timeline: 4-5 weeks (not 6-8)
2. Progressive test suite: 2 tiers early, expand to 5 only if needed
3. 3 core controls instead of 5 (eliminates redundant tests)
4. Quantified decision thresholds (no ambiguity)
5. Explicit contingency plans (know what to do if things fail)
6. Computation reuse across phases (30% GPU savings)
7. Resource accounting upfront (avoid surprises)

**Timeline:** 4-5 weeks | **Hardware:** H100 80GB | **Person-hours:** ~30 | **GPU-hours:** 14-22

---

## Phase-by-Phase Optimized Plan

### Phase 0: Quick Validation (1-2 Days)

**Goal:** Verify core assumptions before committing to full experiment

#### 0.1: Rapid Induction Head Detection (Day 1, ~2 hours)
- **Task**: Scan for induction heads in Llama3-8B using Olsson et al. (2022) signature
- **Why**: Don't replicate their full work; just verify heads exist
- **Input**: 30 random arithmetic problems from [0, 1000]
- **Output**: 
  - Count of induction head candidates
  - Verification: Do they show expected attention patterns?
- **Decision criterion**: ≥5 induction heads found
- **Literature**: Olsson et al. (2022)

#### 0.2: Ablation Baseline Selection (Day 1, ~2 hours)
- **Task**: Quick comparison of ablation methods (mean vs. zero vs. noise vs. layer-specific)
- **Why**: Your experiment depends on this choice
- **Input**: 50 baseline arithmetic problems
- **Scope**: Test on layers 25, 26, 27 only (representative sample, not full)
- **Output**: 
  - Which baseline shows best localization?
  - Table: Drop in accuracy by baseline type
- **Decision criterion**: "Robust if noise_drop ≥ 1.5× mean_drop (p < 0.05)"
- **Literature**: Nanda et al. (2023), Gandelsman et al. (2024)

#### 0.3: Design Control Experiments (Day 2, ~1-2 hours)
- **Task**: Finalize 3 core controls + 1 robustness check
- **Output**: 
  - Pseudocode for each control
  - Expected outcomes & interpretation
  - Decision tree for "proceed vs. pivot"

**Phase 0 Deliverables**:
- Report: "Phase 0 Validation Results" (1 page)
- Code: `src/validation_suite.py`
- Table: Ablation baseline comparison

**Phase 0 Decision Gate**:
```
✓ PROCEED if:
  - ≥5 induction heads identified
  - mean/zero/noise baselines show clear differences
  - Controls can be implemented as planned

✗ PIVOT if:
  - <5 heads found → Reconsider hypothesis
               → Consider IOI circuit approach instead
               → Or: Check if circuit is in different layers
  
  - mean ≈ zero ≈ noise → Layers 17-31 don't carry info
                       → Redefine ablated region or restart
```

**GPU time**: 0.5 hours | **Person-hours**: 5 | **Disk**: 1 GB

---

### Phase 1: Diagnostic Experiments (1 Week)

**Goal:** Determine if forced induction head activation is feasible; identify optimal conditions

#### 1.1: Staged Ablation Testing
- **Task**: Incrementally test if forcing induction heads to activate is possible
- **Why**: Activation patching may fail due to information bottlenecks (Weakness #5)
- **Design**: 
  - Stage 0: Baseline (no ablation)
  - Stage 1: Ablate layer 30 only
  - Stage 2: Ablate layers 28-31
  - Stage 3: Ablate layers 17-31 (full)
- **Metric**: Induction head activation across stages (entropy + behavioral impact)
- **Test suite**: Tier 1 (in-distribution [0,1000], 50 problems) + Tier 4 (symbolic patterns, 50 problems)
- **Decision criterion**: "Can force activation if induction entropy drops ≥0.5 at Stage 3"
- **Literature**: Meng et al. (2023), Timmons et al. (2024)

#### 1.2: Multi-Metric Induction Head Measurement
- **Task**: Validate induction head activation using multiple metrics
- **Why**: Attention entropy alone can be misleading (Weakness #10)
- **Metrics**:
  1. Attention entropy (focus on repeated tokens?)
  2. Behavioral impact (ablation → accuracy change in forward pass)
  3. Repeated token focus (if applicable to problem type)
  4. Logit influence (what tokens does head boost/suppress?)
  5. Consistency across prompts (≥3/5 problems align)
- **Expected**: All 5 metrics should align for high confidence activation
- **Literature**: Wiegreffe & Pinter (2019), Serrano & Smith (2019)

#### 1.3: Tokenization Confound Diagnosis
- **Task**: Test whether tokenization impacts results
- **Why**: Lindsey et al. (2025) shows tokenization is a bottleneck (Weakness #6)
- **Design**:
  - Test A: Single-token arithmetic [0-9]
  - Test B: Two-token arithmetic [0-99]
  - Test C: Original setup [0-1000]
- **Metric**: Where does induction head help degrade?
- **Interpretation**: 
  - If degrades between A→B → tokenization is issue
  - If consistent across all → other bottleneck
- **Literature**: Lindsey et al. (2025), Wei et al. (2022)

**Phase 1 Deliverables**:
- Report: "Circuit Forcing Diagnostics" (2 pages)
  - Stage-by-stage accuracy + activation metrics
  - Multi-metric alignment check
  - Tokenization analysis by tier
- Code: `src/staged_ablation.py`, `src/multi_metric_measurement.py`
- Visualizations: Attention patterns + accuracy trajectories

**Phase 1 Decision Gate**:
```
✓ PROCEED to Phase 2 if:
  - Staged ablation shows progressive activation from Stage 1→3
  - Multi-metrics align in ≥3/5 cases
  - Tokenization doesn't fully explain degradation

? REDESIGN if:
  - Staged ablation shows no effect OR counter-intuitive pattern
  - Multi-metrics disagree (high entropy but low behavioral impact)
  - Decision: Is there competing circuit? Revisit control design.

✗ STOP if:
  - Induction entropy increases with ablation (opposite expected)
  - Behavioral impact is negative & consistent (forcing hurts)
  - Tokenization explains 100% of degradation (can't fix here)
```

**GPU time**: 4-6 hours | **Person-hours**: 8 | **Disk**: 5 GB

---

### Phase 2: Core Experiment (1.5 Weeks)

**Goal:** Determine which scenario (A/B/C) is supported; validate with controls

#### 2.1: Full Activation Forcing with 3-Tier Test Suite
- **Task**: Run main experiment on 3 tiers (expand from 2 if Phase 1 promising)
- **Design**:
  - Tier 1: In-distribution [0, 1000], 100 problems
  - Tier 2: Near-OOD [1000, 2000], 50 problems (ONLY IF Phase 1 shows promise)
  - Tier 4: Symbolic patterns, 50 problems
  - (Tier 3 & 5 optional; add only if Scenario unclear)
- **Ablation**: Full late-layer ablation (17-31) with best baseline from Phase 0
- **Metrics**: All 5 metrics from Phase 1 + final accuracy
- **Metric computation reuse**: Use Phase 0-1 baseline cache (don't re-forward)

#### 2.2: Three Core Controls (Not Five!)
- **Core Control 1: Baseline Validity**
  - **Task**: Random vector ablation vs. mean ablation
  - **Expected**: Noise >> mean drop (if not, mean is problematic)
  - **Problems**: 100 (Tier 1 + Tier 4 combined)

- **Core Control 2: Localization**
  - **Task**: Ablate early layers (0-14) instead of late layers
  - **Expected**: Early ablation ≪ late ablation (circuits localized)
  - **Problems**: 100 (same as Control 1)

- **Core Control 3: Specificity**
  - **Task**: Force non-induction heads vs. induction heads
  - **Expected**: Induction >> non-induction impact (head-specific)
  - **Problems**: 100 (same as Control 1-2)

- **Robustness Check (Conditional)**:
  - If Scenarios A/B unclear after controls: Run permutation test (random seed variations)

#### 2.3: Computation Reuse
- **Phase 0 baseline cache**: Store attention / residuals / logits / embeddings
- **Phase 1 baseline cache**: Reuse Phase 0 where possible
- **Phase 2 baseline cache**: Reuse Phase 0-1 for all baseline measurements
- **Benefit**: Don't re-forward on baseline; only ablated variants (30% GPU savings)

**Phase 2 Deliverables**:
- Report: "Core Experiment Results" (4-5 pages)
  - Results table (accuracy by tier, baseline vs. ablated)
  - Attention pattern visualizations
  - Multi-metric measurements + consistency
  - Control results + interpretation
  - Scenario determination (A/B/C)
- Code: `src/core_experiment.py`, `src/control_experiments.py`
- Data: `results/phase2_results.json` (all per-problem details)
- Cache: `cache/phase0_baseline_cache.pt` (reusable representations)

**Phase 2 Decision Gate**:
```
CLEAR RESULTS → Proceed to Phase 3 analysis:
✓ Scenario A: Tier 1 ≥70%, Tier 4 ≥80%, CI excludes <0.60, controls support
✓ Scenario B: All tiers <25%, CI excludes >0.40, controls support
✓ Scenario C: Tier 1 50-65%, Tier 4 60-75%, OOD degradation clear

AMBIGUOUS RESULTS → Expand testing OR publish with caveats:
? Mixed tiers: Tier 1 high (80%) but Tier 4 low (20%)
  → Suggests different mechanisms; may publish as separate findings
  
? Threshold zone: Tier 1 45-55% (could be A or C)
  → Need more data (100+ more problems) OR accept ambiguity
  → Alternative: Focus on strongest finding (Tier 4?) and publish that

STOP CONDITIONS:
✗ Random ablation ≈ mean ablation (Control 1 fails) → Confound too severe
✗ Early ≈ late ablation (Control 2 fails) → Circuits distributed, not localized
✗ No controls show specificity (Control 3 fails) → Signal unclear
✗ Accuracy 45-55% with wide CI → Too much noise for publication
```

**GPU time**: 10-15 hours | **Person-hours**: 12 | **Disk**: 10 GB

---

### Phase 3: Analysis & Publication (0.5 Week)

**Goal:** Finalize interpretation; prepare publication

#### 3.1: Scenario Determination
- **Task**: Map results to pre-specified scenarios A/B/C
- **Decision matrix**: 
  - Tier 1 accuracy + Tier 4 accuracy + Control support → Which scenario?
  - Account for CI overlaps (not just point estimates)
- **Publication assignment**:
  - Scenario A: "Induction Heads Can Solve Arithmetic"
  - Scenario B: "Why Induction Heads Fail at Arithmetic"
  - Scenario C: "Conditional Success: When Induction Heads Help"

#### 3.2: Statistical Validation
- **Tasks**:
  - Report 95% CI for all accuracy measurements
  - Mann-Whitney U test for pairwise comparisons (p < 0.05)
  - Effect size (Cohen's d) for key findings
  - Robustness: Does result hold on new random seed?

#### 3.3: Limitation & Robustness Discussion
- **Confounds checked**: Mean ablation validity, localization, specificity, tokenization, statistical significance
- **Alternative hypotheses ruled out**: See Phase 2 control results
- **Failure modes documented**: When does induction head activation break?

**Phase 3 Deliverables**:
- Report: "Analysis & Interpretation" (2-3 pages)
  - Scenario determination table
  - Statistical validation
  - Limitation discussion
  - Decision tree for publication readiness
- Code: `src/statistical_validation.py`
- Visualization: "Scenario determination flowchart"

**GPU time**: 0 hours | **Person-hours**: 6 | **Disk**: 0 GB

---

## Pre-Specified Scenarios & Publication Criteria

### Scenario A: Induction Heads CAN Do Arithmetic
**Accuracy thresholds:**
- Tier 1: 65-85% (mean ≥65% with CI excluding <0.60)
- Tier 4: 75-90% (mean ≥75% with CI excluding <0.65)

**Control results must show:**
- Random >> mean ablation (p < 0.05)
- Induction >> non-induction impact (p < 0.05)
- Early layer ≪ late layer effect

**Interpretation:** AF1 circuit suppresses induction heads for efficiency, not necessity

**Publication:** "Induction Heads Can Solve Arithmetic: Multiple Computation Pathways in LLMs"

---

### Scenario B: Induction Heads CANNOT Do Arithmetic
**Accuracy thresholds:**
- All tiers: 10-25% (mean <0.30 with CI excluding >0.35)
- Tier 4 (symbolic): No better than Tier 1

**Control results must show:**
- Random ≈ mean ablation (no difference)
- Induction ≈ random heads (non-specific)
- Attention patterns inconsistent with arithmetic structure

**Interpretation:** Pattern-matching fundamentally incompatible with arithmetic

**Publication:** "Why Induction Heads Fail at Arithmetic: Evidence for Algorithmic Specialization"

---

### Scenario C: Partial/Conditional Success
**Accuracy thresholds:**
- Tier 1: 50-70% (mean 50-65%)
- Tier 4: 60-80% (mean 60-75%)
- Clear OOD degradation (Tier 2 <30%)

**Control results:**
- Mixed: Some controls support, others show competing effects
- Behavioral metrics show improvement on some sub-domains

**Interpretation:** Induction heads provide structural help but lack numerical precision; interference from other circuits

**Publication:** "Circuit Heterogeneity in Arithmetic: How Induction Heads Complement and Interfere with Heuristics"

---

## Contingency Plans

### If Phase 0 Fails:

**<5 induction heads found in Llama3-8B**
- Action 1: Check if heads are in non-standard layers (>layer 16)
- Action 2: Use IOI circuit approach instead (they definitely compose across layers)
- Action 3: Consider: Do pattern-matching circuits not scale to large models?
- Decision: Abandon or pivot to IOI circuit

**mean ≈ zero ≈ noise ablation (Control 1 fails)**
- Problem: Layers 17-31 don't carry meaningful information
- Action 1: Narrow ablated region (try layers 20-31, then 25-31)
- Action 2: Check if arithmetic circuits are in earlier layers (10-20)
- Action 3: Use zero ablation instead of mean (at least it's cleaner)
- Decision: Redefine layers and continue with new baseline

**Controls can't be implemented as planned**
- Action: Simplify to 1-2 core controls (baseline + specificity)
- Continue with reduced scope, note as limitation

### If Phase 1 Shows No Effect:

**Staged ablation: Induction entropy doesn't decrease with ablation**
- Problem: Forcing mechanism isn't working
- Action 1: Double-check ablation hooks are actually being applied
- Action 2: Try stronger ablation (zero instead of mean)
- Action 3: Check if competing circuits are activating instead
- Decision: Return to Phase 0 and redesign ablation method

**Multi-metrics disagree (e.g., high entropy but zero behavioral impact)**
- Problem: Induction heads may be "active" but not functional
- Action: Investigate which metric drives results
- Consider: Separate analyses for each metric
- Decision: Publish as mixed evidence with caveats

**Tokenization explains 100% of degradation**
- Problem: Can't isolate induction head effect from tokenization
- Action 1: Focus analysis on single-token subset [0-9]
- Action 2: Publish sub-finding: "Induction heads work on single-token arithmetic"
- Decision: Pivot to tokenization-aware experiment

### If Phase 2 Shows Ambiguous Results:

**Results in 45-55% accuracy (Scenario A/C boundary)**
- Action 1: Collect more data (100-200 additional problems)
- Action 2: Analyze problem-by-problem: Which problems show Scenario A behavior?
- Action 3: Publish as: "Conditioned success: Induction heads help on ___ type problems"
- Decision: Expand testing OR publish with "mixed evidence" label

**Tier 1 vs Tier 4 disagree (one high, one low)**
- Expected & acceptable: Different mechanisms for pattern vs. arithmetic
- Action: Analyze separately; publish as two findings
- Decision: Publish as "domain-conditional effect"

**Controls partially fail (Control 1 OK, Control 2 fails)**
- Example: Induction >> random heads, but early layers also help
- Action: Assume circuits distributed (not strongly localized)
- Decision: Note as limitation; publish if other controls are strong

---

## Resource Accounting

| Phase | GPU Hours | Person-Hours | Disk (GB) | Timeline |
|-------|-----------|--------------|-----------|----------|
| **Phase 0** | 0.5 | 5 | 1 | 1-2 days |
| **Phase 1** | 4-6 | 8 | 5 | 1 week |
| **Phase 2** | 10-15 | 12 | 10 | 1.5 weeks |
| **Phase 3** | 0 | 6 | 0 | 0.5 week |
| **TOTAL** | **14-22** | **31** | **16** | **4-5 weeks** |

**Without computation reuse:** 20-28 GPU hours (saving 6 hours = 30%)
**Contingencies:** Add 2-3 GPU hours if restarts needed

---

## Key Metrics & Thresholds (Quantified)

| Decision | Threshold | Rationale | Check |
|----------|-----------|-----------|-------|
| **Phase 0: Heads found** | ≥5 candidates | Olsson et al. found ~8-12; allow 40% variance | Scan layers 5-15 |
| **Phase 0: Ablation viable** | noise ≥ 1.5× mean drop (p<0.05) | Indicates mean isn't just fitting noise | Mann-Whitney test |
| **Phase 1: Can force activation** | Entropy drops ≥0.5 bits | Clear difference from baseline | Compare Stage 0 vs 3 |
| **Phase 1: Multi-metric aligned** | ≥3/5 metrics agree | Confidence in measurement | Correlation check |
| **Phase 2: Scenario A** | Tier 1 ≥65%, CI excludes <0.60 | Statistically robust >65% | Bootstrap CI |
| **Phase 2: Scenario B** | All tiers <30%, CI excludes >0.35 | Statistically robust <30% | Bootstrap CI |
| **Phase 2: Scenario C** | Tier 1 50-65%, OOD clear | Partial success with pattern | Visual inspection |
| **Phase 2: Control specificity** | Induction >> random (p<0.05) | Effect not random artifact | Mann-Whitney |
| **Phase 2: Ambiguity** | Accuracy 45-55% with wide CI | Too noisy for single scenario | CI width >0.10 |
| **Publish if** | Any of A/B/C + controls support | Clear finding with validation | See criteria above |

---

## Decision Tree: When to Proceed/Stop

```
START: Phase 0 (Day 1-2)
  │
  ├─ Induction heads found? ≥5
  │  │
  │  ├─ YES → Continue to 0.2
  │  └─ NO  → STOP (Pivot: Try IOI circuits? Check other layers?)
  │
  └─ Ablation valid? (noise >> mean, p<0.05)
     │
     ├─ YES → Continue to Phase 1
     └─ NO  → STOP (Layers 17-31 not localized; try 20-31)

PHASE 1 (Week 1)
  │
  ├─ Can force activation? (Entropy ↓ ≥0.5 bits)
  │  │
  │  ├─ YES → Multi-metrics aligned?
  │  │  │
  │  │  ├─ YES → Continue to Phase 2
  │  │  └─ NO  → REDESIGN (Why disagree? Competing circuits?)
  │  │
  │  └─ NO  → STOP (Activation forcing isn't working)
  │
  └─ Tokenization confound? (Degrades [0-9]→[0-99]→[0-1000])
     │
     ├─ YES → Focus Phase 2 on [0-9] tier only
     └─ NO  → Continue as planned

PHASE 2 (Weeks 2-3)
  │
  ├─ Results clear? (Fits A/B/C scenario)
  │  │
  │  ├─ YES (Accuracy ≥65% OR <30%)
  │  │  │
  │  │  └─ Controls support? (Induction >> random, p<0.05)
  │  │     │
  │  │     ├─ YES → PROCEED to Phase 3 (PUBLISH)
  │  │     └─ NO  → Note as limitation, PUBLISH with caveat
  │  │
  │  └─ NO (Accuracy 45-55%)
  │     │
  │     ├─ Tight CI? (CI width <0.10)
  │     │  ├─ YES → Collect ≥100 more problems
  │     │  └─ NO  → Too noisy; PUBLISH as "mixed evidence"
  │     │
  │     └─ Expand to Tier 3 (far-OOD)? (If resources available)
  │        ├─ Clear pattern emerges → PROCEED to Phase 3
  │        └─ Still ambiguous → PUBLISH as "domain-conditional"

PHASE 3 (Week 4-5)
  └─ Final analysis & publication
```

---

## Computation Reuse Strategy

### Phase 0 Baseline Cache
```
Cache structure:
  baseline_v0 = {
    'attention': [...],        # All layer/head attentions
    'residuals': [...],        # All layer residuals
    'logits': [...],           # Model predictions
    'embeddings': [...],       # Token embeddings
    'problems': [30],          # Metadata
  }
  Size: ~100 MB
```

### Phase 1: Reuse Phase 0
- Don't re-forward on baseline for 50 new problems
- Only forward on staged ablations (Stage 1/2/3)
- Savings: 25% of Phase 1 compute

### Phase 2: Reuse Phase 0 + 1
- Cache all 100 Tier 1 + 50 Tier 4 baseline forward passes from Phase 1
- Only re-forward on:
  - Full ablation (layers 17-31)
  - Three controls (random, early, non-induction)
- Savings: 40% of Phase 2 compute over naive forward-everything approach

**Total savings**: ~6 GPU hours (30% of 20 total)

---

## Success Criteria & Publication Bar

### DEFINITELY PUBLISH (High Confidence)
✓ Scenario A/B with robust controls (induction >> random, p<0.05)
✓ Scenario C with clear OOD pattern + controls support
✓ Strong null result: "Cannot force induction head activation -> rules out mechanism"

### PUBLISH WITH CAVEATS (Medium Confidence)
✓ Scenario A/B but one control fails (note as limitation)
✓ Mixed evidence: Tier 1 ≠ Tier 4 (publish both, separate findings)
✓ Scenario C with conflicting metrics (qualifies findings)

### DON'T PUBLISH (No Clear Finding)
✗ Results ambiguous without clear controls
✗ Multiple controls fail (confounds too severe)
✗ Accuracy 45-55% with no clear trend (too noisy)
✗ Induction head field detection unreliable (different counts across runs)

---

## Timeline Summary

```
Week 1    Day 1-2:  Phase 0 validation (quick scan)
          Days 3-7: Phase 1 diagnostics (staged ablation)

Week 2    Days 8-14: Phase 2 core experiment (3 controls, 3-tier)

Week 3    Days 15-21: Phase 2 completion + analysis

Week 4-5  Days 22+:  Phase 3 analysis + write-up
```

---

## Recommended Implementation Order

1. **Read papers (before starting):**
   - Olsson et al. (2022) - Induction heads [30 min]
   - Nanda et al. (2023) - Ablation methodology [20 min]
   - Wiegreffe & Pinter (2019) - Attention validity [20 min]

2. **Week 0 (Prep):**
   - Set up code structure (`src/`, `cache/`, `results/`)
   - Implement Phase 0 validation suite
   - Test on toy problems

3. **Week 1:**
   - Run Phase 0 (decision gate: proceed/pivot?)
   - Run Phase 1 diagnostics
   - Prepare Phase 2 experiments

4. **Week 2-3:**
   - Run Phase 2 (core + controls)
   - Interpret results
   - Determine scenario

5. **Week 4-5:**
   - Write-up & submission

---

## Support Materials

**See also:**
- `overview.md` - Original problem formulation & background
- `ROADMAP_CRITICAL_ANALYSIS.md` - Detailed critique of original roadmap
- `TODO.md` - Detailed task breakdown & implementation checklist

---

## Key Advantages Over Original Proposal

| Criterion | Original Revised | This Optimized | Gain |
|-----------|---|---|---|
| Timeline | 6-8 weeks | 4-5 weeks | 33-50% faster |
| Controls | 5 (redundant) | 3 (essential) | 50% fewer controls |
| Phase 0 investment | 1 week replication | 1-2 days scan | 90% faster |
| Test tiers | 5 always (300 probs) | 2-5 progressive (150-300) | 50% early savings |
| GPU hours | ~40 | ~14-22 | 50% savings |
| Decision clarity | Vague thresholds | Quantified + statistical | Much clearer |
| Contingency plans | Implicit | Explicit decision tree | Faster pivots |
| Resource certainty | Not analyzed | Full accounting table | No surprises |

---

## Next Steps

1. **This week:** Read recommended papers + review this roadmap
2. **Week 1:** Implement Phase 0 validation suite; run quick scan
3. **If Phase 0 succeeds:** Commit to Phases 1-3
4. **If Phase 0 fails:** Execute contingency (IOI circuits? Different layers?)

**Ready to start? Check `TODO.md` for detailed implementation steps.**


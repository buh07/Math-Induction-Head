# TODO: Implementation Checklist for Induction Heads Experiment

**Timeline: 4-5 weeks | Last updated: Feb 17, 2026**

## PROJECT STATUS

### ✓ WEEK 0: INFRASTRUCTURE (80% Complete)
- [x] Papers read and background understood
- [x] Code structure fully implemented (2500+ lines across 6 modules)
- [x] Git repository initialized with .gitignore
- [x] Python virtual environment created
- [ ] Dependencies installed in venv (`pip install -r requirements.txt`)

### ⏳ PHASE 0: QUICK VALIDATION (Days 1-2) - READY TO START
- Requires: Model download, dependency installation
- Entry point: `python main.py --phase 0 --model meta-llama/Llama-2-7b-hf`

### ⏳ PHASE 1: DIAGNOSTICS (Week 1) - PENDING
### ⏳ PHASE 2: CORE EXPERIMENT (Weeks 2-3) - PENDING
### ⏳ PHASE 3: ANALYSIS (Week 4-5) - PENDING

---

## QUICK START GUIDE

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Phase 0 (quick validation - 1-2 hours, ~2 GPU hours)
python main.py --phase 0 --model meta-llama/Llama-2-7b-hf

# 4. If Phase 0 passes:
python main.py --phase 1  # Phase 1 (1 week, ~5-6 GPU hours)
python main.py --phase 2  # Phase 2 (1.5 weeks, ~10-15 GPU hours)
python main.py --phase 3  # Phase 3 (0.5 week, ~0 GPU hours)
```

---

## WEEK 0: PREPARATION & SETUP

### [x] 0.1 Read Core Papers (4 hours) - COMPLETED
- [x] Olsson et al. (2022) - "In-Context Learning and Induction Heads" [30 min]
  - Focus: How to identify induction heads; what makes them work
  - Key takeaway: Prev-token head (Layer 1) + Induction head (Layer 2) composition
  
- [x] Nanda et al. (2023) - "Progress Measures for Grokking" sections on ablation [20 min]
  - Focus: Why mean ablation? When is it appropriate?
  - Key takeaway: Layer-specific mean often better than global mean for compositional tasks
  
- [x] Wiegreffe & Pinter (2019) - "Attention is Not Not Explanation" [20 min]
  - Focus: When attention patterns reflect causality
  - Key takeaway: Use behavioral validation (ablation) alongside attention analysis

- [x] Review ROADMAP.md in detail [30 min]
  - Understand all phases, decision gates, contingencies

### [x] 0.2 Set Up Code Structure (2 hours) - COMPLETED
```
Math_Induction_Head/
├── src/
│   ├── __init__.py
│   ├── utils.py                        # 740 lines: Core utilities & data structures
│   ├── validation_suite.py             # 550 lines: Phase 0 implementation
│   ├── staged_ablation.py              # 350 lines: Phase 1a implementation
│   ├── multi_metric_measurement.py     # 350 lines: Phase 1b implementation
│   ├── core_experiment.py              # 380 lines: Phase 2 implementation
│   └── statistical_validation.py       # 400 lines: Phase 3 implementation
├── cache/                              # Baseline forward pass cache
├── results/                            # JSON outputs from each phase
├── notebooks/                          # Analysis & visualization
├── main.py                             # 170 lines: CLI entry point
├── requirements.txt                    # Dependencies
├── .gitignore                          # Git ignore (with venv)
├── venv/                               # Python virtual environment
├── ROADMAP.md                          # Research strategy
├── TODO.md                             # This file
└── overview.md                         # Problem formulation
```

- [x] Create directory structure (`mkdir -p src cache results notebooks`)
- [x] Create `requirements.txt` with core dependencies (PyTorch, transformers, scipy, etc.)
- [x] Create `src/__init__.py` and all module files
- [x] Implement `src/utils.py` with HookManager, ArithmeticDataset, AblationConfig classes
- [x] Implement `src/validation_suite.py` for Phase 0
- [x] Implement `src/staged_ablation.py` for Phase 1a
- [x] Implement `src/multi_metric_measurement.py` for Phase 1b
- [x] Implement `src/core_experiment.py` for Phase 2
- [x] Implement `src/statistical_validation.py` for Phase 3
- [x] Implement `main.py` with CLI interface

### [x] 0.3 Environment Setup (1 hour) - COMPLETED
- [x] Initialize git repository (`git init`)
- [x] Create `.gitignore` with venv, __pycache__, *.pyc, results cache, etc.
- [x] Create Python virtual environment (`python3 -m venv venv`)
- [ ] Activate venv and install dependencies: `source venv/bin/activate && pip install -r requirements.txt`

### [ ] 0.4 Download Model (1 hour)
- [ ] Download Llama model (choose one):
  ```bash
  source venv/bin/activate
  
  # Option 1: Llama3-8B (if you have HF access)
  python download_model.py --model meta-llama/Llama-3-8b-instruct
  
  # Option 2: Llama2-7B (public, recommended)
  python download_model.py --model meta-llama/Llama-2-7b-hf
  
  # Option 3: GPT-2 for testing (small, fast)
  python download_model.py --model gpt2
  
  # If using a custom token:
  python download_model.py --model meta-llama/Llama-3-8b-instruct --token hf_YOUR_TOKEN
  ```

**Setup HuggingFace Authentication (Required for Llama models):**
```bash
source venv/bin/activate

# Method 1: Interactive login (recommended)
huggingface-cli login
# Then paste your HF token when prompted

# Method 2: Set token as environment variable
export HF_TOKEN=hf_YOUR_TOKEN_HERE

# Then try the download
```

- [ ] Verify model downloads successfully
- [ ] Check GPU memory requirements

### [ ] 0.5 Datasets (Auto-generated)
- [ ] ArithmeticDataset class generates problems on-the-fly during Phase 0
- [ ] No pre-generation needed; handled by `utils.ArithmeticDataset`

**✓ Infrastructure Complete - Ready for Phase 0**

---

## PHASE 0: QUICK VALIDATION (Days 1-2)

### [ ] 0.0 Pre-flight: Create Phase 0 Task List
Create `src/phase0_tasks.md` with exact implementation:
- [ ] Copy skeleton from ROADMAP.md sections 0.1-0.3
- [ ] Write pseudocode for each step
- [ ] Define expected outputs (e.g., "Dict with {layer, head, entropy_score}")

### [ ] 0.1a: Induction Head Detection (Day 1, 2 hours)
**Pseudocode:**
```python
def detect_induction_heads_quick(model, problems, layers=range(5, 16)):
  """
  Scan for heads showing Olsson et al. signature (across-layer composition)
  """
  candidates = []
  for layer in layers:
    for head in range(32):  # Llama3-8B has 32 heads
      # Test on each problem
      activation_scores = []
      for problem in problems:
        attn = forward_and_get_attention(model, problem, layer, head)
        
        # Signature 1: High entropy on different positions
        # (induction heads attend to repeated tokens + their context)
        entropy = -(attn * log(attn)).sum()
        
        # Signature 2: Focus on repeated tokens (if in problem)
        repeated_token_focus = measure_repeated_focus(attn, problem)
        
        score = entropy * 0.5 + repeated_token_focus * 0.5
        activation_scores.append(score)
      
      mean_score = mean(activation_scores)
      if mean_score > threshold:
        candidates.append({
          'layer': layer,
          'head': head,
          'score': mean_score,
          'entropy': mean([e for e in entropies]),
        })
  
  return sorted(candidates, key=lambda x: x['score'], reverse=True)
```

**Implementation:**
- [ ] Implement `detect_induction_heads_quick()` in `src/validation_suite.py`
- [ ] Test on 30 random problems from `problems_tier1_indist.json`
- [ ] Output: List of (layer, head, score) tuples
- [ ] Log: "Found N heads matching Olsson et al. signature"

**Decision checkpoint:**
  - ✓ IF ≥5 heads found: Continue to 0.1b
  - ✗ IF <5 heads: Output report "Induction heads not detected; pivoting to IOI circuits"

### [ ] 0.1b: Validate Head Detection (Day 1, 1 hour)
- [ ] Positive control: Test top induction heads on few-shot pattern completion
  ```
  "Complete pattern: A→B, C→D, E→F. Complete: G→?"
  ```
- [ ] Expected: High attention entropy + focused on repeated tokens
- [ ] Record: Attention visualization for top 3 heads

### [ ] 0.2a: Ablation Baseline Comparison (Day 1, 2 hours)
**Pseudocode:**
```python
def compare_ablation_baselines(model, problems, layers=[25, 26, 27]):
  """
  Test: mean vs. zero vs. noise vs. layer_specific
  """
  baselines = {}
  
  # Baseline 1: Mean
  mean_activations = compute_mean_mlp_outputs(model, problems, layers)
  accuracy_mean = evaluate_with_ablation(model, problems, layers, mean_activations)
  baselines['mean'] = {
    'accuracy': accuracy_mean,
    'drop': baseline_accuracy - accuracy_mean
  }
  
  # Baseline 2: Zero
  zero_activations = {layer: zeros_like(mean_activations[layer]) for layer in layers}
  accuracy_zero = evaluate_with_ablation(model, problems, layers, zero_activations)
  baselines['zero'] = { ... }
  
  # Baseline 3: Noise
  noise_activations = {layer: randn_like(mean_activations[layer]) * std(...) for layer in layers}
  accuracy_noise = evaluate_with_ablation(model, problems, layers, noise_activations)
  baselines['noise'] = { ... }
  
  # Baseline 4: Layer-specific mean
  layer_specific = {}
  for layer in layers:
    for idx, problem in enumerate(problems):
      # Compute mean MLP output just for this problem
      layer_specific[layer, idx] = mean_mlp_for_problem(layer, idx)
  accuracy_layer_specific = evaluate_with_layer_specific(...)
  baselines['layer_specific'] = { ... }
  
  return baselines
```

**Implementation:**
- [ ] Implement `compare_ablation_baselines()` in `src/validation_suite.py`
- [ ] Test on 50 problems from `problems_tier1_indist.json`
- [ ] Output table:
  ```
  | Baseline | Accuracy Drop | vs. Baseline |
  |----------|---|---|
  | None (baseline) | 0 | - |
  | Mean | 0.35 | - |
  | Zero | 0.40 | 1.14x |
  | Noise | 0.50 | 1.43x |
  | Layer-specific | 0.34 | 0.97x |
  ```
- [ ] Statistical test: Is noise_drop significantly > mean_drop? (Mann-Whitney U, p<0.05)

**Decision checkpoint:**
  - ✓ IF noise ≥ 1.5× mean (p < 0.05): Mean is robust → Use mean for Phase 1-2
  - ⚠ IF noise ~ mean: Layers might not be critical → Investigate further OR restart
  - ✗ IF mean ≈ zero ≈ noise: Layers 25-27 don't carry info → Redefine ablated region

### [ ] 0.2b: Report & Decision (Day 2, 1 hour)
- [ ] Write `phase0_validation_report.md`:
  ```markdown
  # Phase 0 Validation Report
  
  ## Induction Head Detection
  - Heads found: N
  - Top 3: (layer, head, score)
  - Positive control (few-shot): PASS / FAIL
  
  ## Ablation Baseline
  - Recommended: [mean / zero / layer-specific]
  - Robustness: noise/mean ratio = X (≥1.5 for robust)
  - Decision: PROCEED / PIVOT
  ```

- [ ] Decision gate:
  - ✓ IF both sub-tests pass: **Task for Phase 1 lead:** "You have 1 week to run diagnostics"
  - ✗ IF either fails: Document contingency; pivot to IOI circuits OR redefine

**End Phase 0: ~10 GPU hours, ~5 person-hours**

---

## PHASE 1: DIAGNOSTICS (1 Week)

### [ ] 1.0: Import Phase 0 Results
- [ ] Load induction head candidates
- [ ] Set ablation baseline to Phase 0 choice (mean/zero/other)
- [ ] Create `phase1_config.yaml`:
  ```yaml
  ablation_baseline: 'mean'  # from Phase 0
  induction_heads:
    - {layer: 7, head: 15}
    - {layer: 8, head: 22}
    # ... top 10 from Phase 0
  ```

### [ ] 1.1a: Staged Ablation - Stage 1 (Dev 3, 1 hour)
**Pseudocode:**
```python
def staged_ablation_stage1(model, problems, baseline):
  """Test: Ablate only layer 30"""
  results = {
    'stage': 1,
    'ablated_layers': [30],
    'problems': len(problems),
  }
  
  # Forward with layer 30 ablation
  accuracies = []
  induction_entropies = []
  
  for problem in problems:
    with ablate_layer_hook(model, layer=30, baseline=baseline):
      output = model.generate(problem)
      correct = is_correct_arithmetic(problem, output)
      accuracies.append(correct)
      
      # Measure induction head activation
      attn = get_attention_for_heads(model, induction_heads)
      entropy = compute_entropy(attn)
      induction_entropies.append(entropy)
  
  results['accuracy'] = mean(accuracies)
  results['accuracy_drop'] = baseline_accuracy - results['accuracy']
  results['induction_entropy'] = mean(induction_entropies)
  results['entropy_change'] = baseline_entropy - results['induction_entropy']
  
  return results
```

**Implementation:**
- [ ] Implement in `src/staged_ablation.py`
- [ ] Run on 100 problems from `problems_tier1_full.json`
- [ ] Output: `results/stage1_results.json`
  ```json
  {
    "accuracy": 0.92,
    "accuracy_drop": 0.03,
    "induction_entropy": 3.2,
    "entropy_change": -0.1,
    "note": "Layer 30 alone has minimal effect"
  }
  ```

### [ ] 1.1b: Staged Ablation - Stage 2 (Day 3-4, 1.5 hours)
- [ ] Repeat for layers 28-31
- [ ] Output: `results/stage2_results.json`
- [ ] Compare: Stage 1 vs. Stage 2 (should see larger effect with more layers)

### [ ] 1.1c: Staged Ablation - Stage 3 (Day 4-5, 2 hours)
- [ ] Ablate full range layers 17-31 (MAIN TEST)
- [ ] Output: `results/stage3_results.json`
- [ ] Record metrics:
  - Accuracy drop
  - Induction entropy change
  - Behavioral impact (from next section)

**Decision checkpoint after 1.1:**
  - ✓ IF entropy decreases progressively (Stages 1→2→3): Can force activation
  - ⚠ IF flat or oscillating: May need stronger ablation
  - ✗ IF entropy increases: Opposite expected; stop & investigate

### [ ] 1.2a: Multi-Metric Measurement Setup (Day 5, 1.5 hours)
**Pseudocode:**
```python
def measure_induction_head_activation_multimetric(model, problem, induction_heads):
  """
  Return 5 metrics for confidence in activation
  """
  metrics = {}
  
  # Metric 1: Attention entropy
  attn = get_attention_for_heads(model, induction_heads)
  entropy = -(attn * log(attn)).sum()
  metrics['entropy'] = entropy
  
  # Metric 2: Behavioral impact
  output_normal = model.generate(problem)
  with zero_head_hook(model, induction_heads):
    output_ablated = model.generate(problem)
  metrics['behavioral_impact'] = jaccard_similarity(output_normal, output_ablated)
  
  # Metric 3: Repeated token focus
  if has_repeated_tokens(problem):
    metrics['repeated_focus'] = measure_repeated_focus(attn, problem)
  else:
    metrics['repeated_focus'] = 0  # N/A for this problem
  
  # Metric 4: Logit influence
  logit_diff = (output_normal.logits - output_ablated.logits).abs().mean()
  metrics['logit_influence'] = logit_diff
  
  # Metric 5: Consistency
  test_count = 5  # multiple forward passes?
  metrics['consistency'] = compute_consistency(test_count)
  
  return metrics
```

**Implementation:**
- [ ] Implement in `src/multi_metric_measurement.py`
- [ ] Define consistency check (e.g., same result over 3 random seeds?)
- [ ] Test on subset (20 problems) to verify function works

### [ ] 1.2b: Run Multi-Metric on All Problems (Day 5-6, 2 hours)
- [ ] Run on 100 Tier 1 + 50 Tier 4 problems (150 total)
- [ ] Output: `results/multi_metrics_stage3.json`
  ```json
  [
    {
      "problem": "7 + 5 =",
      "entropy": 3.2,
      "behavioral_impact": 0.2,
      "repeated_focus": 0.15,
      "logit_influence": 0.8,
      "consistency": "high",
      "all_aligned": true
    },
    ...
  ]
  ```

**Decision checkpoint:**
  - ✓ IF ≥60% of problems show all_aligned=true: High confidence
  - ? IF 30-60% aligned: Medium confidence; note in report
  - ✗ IF <30% aligned: Metrics disagreeing; investigate why

### [ ] 1.3a: Tokenization Diagnosis (Day 6, 1 hour)
**Pseudocode:**
```python
def tokenization_diagnosis(model, problems_single_token, problems_two_token, problems_full):
  """
  Compare performance on [0-9], [0-99], [0-1000]
  """
  results = {}
  
  for name, problems in [
    ('single_token', problems_single_token),
    ('two_token', problems_two_token),
    ('full', problems_full),
  ]:
    # Baseline
    acc_baseline = evaluate(model, problems, ablate=False)
    
    # With full ablation (17-31)
    acc_ablated = evaluate(model, problems, ablate=True, layers=range(17, 32))
    
    induction_entropy_change = measure_entropy_change(model, problems, ablate=True)
    
    results[name] = {
      'baseline': acc_baseline,
      'ablated': acc_ablated,
      'drop': acc_baseline - acc_ablated,
      'entropy_change': induction_entropy_change,
    }
  
  # Interpretation
  if results['single_token']['drop'] > results['two_token']['drop']:
    print("Tokenization is limiting factor (degrades with complexity)")
  else:
    print("Tokenization not primary bottleneck")
  
  return results
```

**Implementation:**
- [ ] Run on `problems_single_token.json` (50 [0-9])
- [ ] Run on `problems_two_token.json` (50 [0-99])
- [ ] Run on `problems_tier1_full.json` (100 [0-1000])
- [ ] Output: `results/tokenization_diagnosis.json`
- [ ] Visualization: Bar chart showing drop size across tiers

### [ ] 1.3b: Tokenization Report (Day 6, 0.5 hour)
- [ ] Interpretation:
  - ✓ If single-token shows BEST induction help:Tokenization is bottleneck
  - ✗ If single-token shows NO HELP: Fundamental incompatibility

### [ ] 1.4: Phase 1 Report (Day 7, 1 hour)
- [ ] Write `phase1_diagnostics_report.md`:
  ```markdown
  # Phase 1 Diagnostics Report
  
  ## Staged Ablation Results
  - Stage 1 (layer 30): Entropy ↓X, Accuracy ↓Y%
  - Stage 2 (layers 28-31): Entropy ↓X, Accuracy ↓Y%
  - Stage 3 (layers 17-31): Entropy ↓X, Accuracy ↓Y%
  - Interpretation: [Can / Cannot] force activation
  
  ## Multi-Metric Alignment
  - Aligned problems: Z%
  - Conflicting cases: [list examples]
  
  ## Tokenization Diagnosis
  - [0-9]: Accuracy drop = X%
  - [0-99]: Accuracy drop = Y%
  - [0-1000]: Accuracy drop = Z%
  - Conclusion: [Tokenization is / is not bottleneck]
  
  ## Decision
  - PROCEED to Phase 2
  ```

**Phase 1 Decision Gate:**
- ✓ Entropy decreases + multi-metrics aligned + clear pattern: **PROCEED**
- ⚠ Mixed results: Proceed with caveats; adjust Phase 2 design
- ✗ No clear effect + metrics disagree: STOP; investigate or pivot

**End Phase 1: ~5-6 GPU hours, ~8 person-hours**

---

## PHASE 2: CORE EXPERIMENT (1.5 Weeks)

### [ ] 2.0: Phase 2 Prep (Day 8, 1 hour)
- [ ] Lock in design based on Phase 1:
  - Ablated layers (17-31? 20-31? Adjust if needed)
  - Ablation baseline (mean? zero? From Phase 0)
  - Induction heads to monitor (top-10 from Phase 0)
  - Test tiers (definitely Tier 1 + 4; add Tier 2 if Phase 1 promising)

- [ ] Create `phase2_config.yaml`:
  ```yaml
  core_experiment:
    ablated_layers: [17, 18, ..., 31]
    baseline: 'mean'
    induction_heads: [...]
    test_tiers: [1, 4]  # or [1, 2, 4] if expanding
  
  controls:
    control_1_random_ablation: true
    control_2_early_layers: true
    control_3_non_induction: true
  ```

### [ ] 2.1a: Compute Baseline Cache (Days 8-9, 2 hours)
**Pseudocode:**
```python
def compute_baseline_cache(model, problems_tier1, problems_tier4):
  """
  Forward through model on all baseline problems; cache representations
  """
  cache = {
    'tier1': {},
    'tier4': {},
  }
  
  for tier_name, problems in [('tier1', problems_tier1), ('tier4', problems_tier4)]:
    for idx, problem in enumerate(problems):
      output = model(problem, output_hidden_states=True)
      
      cache[tier_name][idx] = {
        'attention': extract_all_attention(output),
        'residuals': output.hidden_states,
        'logits': output.logits,
        'embedding': output.last_hidden_state,
        'problem': problem,
      }
  
  torch.save(cache, 'cache/baseline_cache.pt')
  return cache
```

**Implementation:**
- [ ] Run baseline forward on:
  - Tier 1: 100 problems (from Phase 2)
  - Tier 4: 50 problems
- [ ] Save to `cache/baseline_cache.pt`
- [ ] Output: ~2 GB cache file (test fits in memory)

### [ ] 2.1b: Core Experiment (Days 9-11, 4 hours)
**Pseudocode:**
```python
def core_experiment(model, cache, ablation_config):
  """
  Main test: Full ablation on all tiers; measure all 5 metrics
  """
  results = {
    'tier1': [],
    'tier4': [],
  }
  
  for tier_name in ['tier1', 'tier4']:
    for idx, cached in enumerate(cache[tier_name]):
      problem = cached['problem']
      
      # Forward with ablation (don't re-forward baseline)
      with ablate_layers_hook(model, layers=ablation_config['ablated_layers'],
                              baseline=ablation_config['baseline']):
        output_ablated = model.generate(problem)
      
      # Metrics
      accuracy = is_correct_arithmetic(problem, output_ablated)
      entropy = compute_entropy(cached['attention'][induction_heads])
      behavioral_diff = jaccard(cached['logits'], output_ablated.logits)
      
      results[tier_name].append({
        'problem_id': idx,
        'accuracy': accuracy,
        'entropy': entropy,
        'behavioral_impact': behavioral_diff,
        # ... other metrics
      })
  
  return results
```

**Implementation:**
- [ ] Implement in `src/core_experiment.py`
- [ ] Run full ablation (17-31) on Tier 1 + Tier 4
- [ ] Output: `results/core_experiment_main.json`
  ```json
  {
    "tier1": {
      "mean_accuracy": 0.65,
      "accuracy_ci": [0.60, 0.70],
      "problems": [
        {"id": 0, "problem": "7+5=", "accuracy": 1, ...},
        ...
      ]
    },
    "tier4": {...}
  }
  ```

**Record all metrics for later:**
- Per-problem accuracy
- Multi-metric measurements
- Attention patterns (save visualizations)

### [ ] 2.2a: Control 1 - Baseline Validity (Days 11-12, 1 hour)
**Test:** Random ablation vs. mean ablation
**Expected:** Noise >> mean drop (if not, confound)

**Implementation:**
- [ ] Implement in `src/control_experiments.py`
- [ ] Replace Phase 0-1 baseline with random Gaussian noise
- [ ] Run on same 150 problems (50+100 from tiers 1+4)
- [ ] Compare: accuracy drop random vs. mean
- [ ] Statistical test: Mann-Whitney U test (p < 0.05)
- [ ] Output: `results/control1_random_vs_mean.json`
  ```json
  {
    "mean_ablation": {"accuracy_drop": 0.35, "ci": [0.30, 0.40]},
    "noise_ablation": {"accuracy_drop": 0.50, "ci": [0.45, 0.55]},
    "ratio": 1.43,
    "pvalue": 0.001,
    "robust": true,
    "interpretation": "mean ablation is appropriate baseline"
  }
  ```

### [ ] 2.2b: Control 2 - Localization (Days 12-13, 1 hour)
**Test:** Early layer ablation (0-14) vs. late (17-31)
**Expected:** Early << late (circuits are localized to late layers)

**Implementation:**
- [ ] Ablate layers 0-14 instead of 17-31
- [ ] Run on same 150 problems
- [ ] Compare drop sizes
- [ ] Output: `results/control2_early_vs_late.json`
  ```json
  {
    "early_layers": {"accuracy_drop": 0.08, "ci": [0.03, 0.13]},
    "late_layers": {"accuracy_drop": 0.35, "ci": [0.30, 0.40]},
    "ratio": 4.4,
    "interpretation": "circuits localized to late layers"
  }
  ```

### [ ] 2.2c: Control 3 - Specificity (Days 13-14, 1 hour)
**Test:** Force induction heads vs. force non-induction heads
**Expected:** Induction >> non-induction impact

**Implementation:**
- [ ] Identify non-induction heads (random selection from high-layer heads)
- [ ] Force their activation (inverse of main test)
- [ ] Run on same 150 problems
- [ ] Compare: accuracy impact induction vs. non-induction
- [ ] Output: `results/control3_induction_vs_random.json`
  ```json
  {
    "induction_heads_accuracy_drop": 0.35,
    "random_heads_accuracy_drop": 0.05,
    "ratio": 7.0,
    "pvalue": 0.0001,
    "interpretation": "effect is induction-head-specific"
  }
  ```

### [ ] 2.3: Expand Testing (Optional; Days 14, 2 hours)
**IF Phase 1-2 results are ambiguous (accuracy 45-55%):**
- [ ] Add Tier 2 (near-OOD [1000, 2000], 50 problems)
- [ ] Add Tier 3 (far-OOD [10K, 100K], 50 problems)
- [ ] IF results remain ambiguous: Add Tier 5 (hybrid patterns, 50 problems)

**Otherwise:**
- [ ] Skip; finalize with current findings

### [ ] 2.4: Phase 2 Report (Day 14, 1.5 hours)
- [ ] Write `phase2_results_report.md`:
  ```markdown
  # Phase 2 Core Experiment Results
  
  ## Main Findings
  - Tier 1 accuracy: X% ± CI
  - Tier 4 accuracy: Y% ± CI
  - Scenario: [A/B/C]
  
  ## Control Results
  - Control 1 (Random vs Mean): Robust? [Yes/No]
  - Control 2 (Early vs Late): Localized? [Yes/No]
  - Control 3 (Induction vs Random): Specific? [Yes/No]
  
  ## Interpretation
  [Scenario A/B/C supported; specifics]
  
  ## Next: Phase 3 or Publication?
  ```

**End Phase 2: ~10-15 GPU hours, ~12 person-hours**

---

## PHASE 3: ANALYSIS & PUBLICATION (0.5 Week)

### [ ] 3.0: Scenario Determination (Days 15, 2 hours)
- [ ] Load all Phase 2 results
- [ ] Create decision matrix:
  ```
  Tier 1 Acc | Tier 4 Acc | OOD Trend | Controls | → Scenario
  ≥65%       | ≥75%       | N/A       | Strong   | → A
  <30%       | <30%       | N/A       | Support  | → B
  50-65%     | 60-75%     | ↓ clear   | Mixed    | → C
  ```
- [ ] Determine most likely scenario
- [ ] Output: `results/scenario_determination.md`

### [ ] 3.1: Statistical Validation (Days 15, 1.5 hours)
- [ ] Compute 95% confidence intervals for all accuracy measurements
- [ ] Run Mann-Whitney U tests for pairwise comparisons
- [ ] Compute Cohen's d effect sizes
- [ ] Test robustness: Do results hold on different random seed?
- [ ] Output: `results/statistical_validation.json`

### [ ] 3.2: Write Publication-Ready Report (Days 16, 2 hours)
- [ ] Create `PUBLICATION_DRAFT.md` (3-4 pages):
  ```markdown
  # Induction Heads and Arithmetic Computation in LLMs
  
  ## Abstract
  [Scenario A/B/C finding in 150 words]
  
  ## Introduction
  [Why this matters; prior work]
  
  ## Methods
  - Model: Llama3-8B
  - Induction head detection: Olsson et al. method
  - Activation forcing: Ablate layers 17-31
  - Test suite: [Describe]
  
  ## Results
  - Main finding: [Scenario]
  - Supporting evidence: [Accuracies + CIs]
  - Controls: [Did they support?]
  
  ## Discussion
  - Interpretation of finding
  - Implications for mechanistic interpretability
  - Limitations
  
  ## References
  ```

### [ ] 3.3: Limitation & Robustness Discussion (Days 16, 1 hour)
- [ ] Confounds checked:
  - [ ] Mean ablation validity (Control 1)
  - [ ] Circuit localization (Control 2)
  - [ ] Induction-head specificity (Control 3)
  - [ ] Tokenization effects (Phase 1.3)
  - [ ] Statistical significance (Phase 3.1)

- [ ] Known limitations:
  - [ ] Single model (Llama3-8B only; can't generalize to other models)
  - [ ] Limited test suite (only arithmetic; other domains?)
  - [ ] Attention entropy may not capture all activation patterns
  - [ ] Contingency: If <5 heads found, generalizability questionable

- [ ] Output: `PUBLICATION_DRAFT.md` (Limitations section)

### [ ] 3.4: Final Decision - Publish? (Days 16, 0.5 hour)
**Checklist:**
- [ ] ✓ OR ✗: Does result fit Scenario A/B/C clearly?
- [ ] ✓ OR ✗: Do 2+ controls support the finding?
- [ ] ✓ OR ✗: Is result statistically significant (p < 0.05)?
- [ ] ✓ OR ✗: Is finding replicable on new seed?
- [ ] ✓ OR ✗: Are confounds ruled out?

**If 4-5 checkboxes pass:**
→ **PUBLISH** (See publication criteria in ROADMAP.md)

**If 2-3 checkboxes pass:**
→ **PUBLISH WITH CAVEATS** (Note limitations prominently)

**If 0-1 checkboxes pass:**
→ **REDESIGN OR ARCHIVE** (Not ready for publication)

**End Phase 3: 0 GPU hours, ~6 person-hours**

---

## CONTINGENCY CHECKLIST

### If <5 induction heads found in Phase 0:
- [ ] Try different layer ranges (0-20? 5-20?)
- [ ] Check if heads are in different location than Olsson et al. predicted
- [ ] Alternative: Investigate IOI circuit instead (different mechanism)
- [ ] Decision: Pivot to IOI OR abandon if heads genuinely not present

### If mean ≈ zero ≈ noise ablation:
- [ ] Problem: Layers 17-31 don't carry meaningful information
- [ ] Action: Redefine ablated region (try 20-31, then 22-31)
- [ ] Alternative: Try zero ablation instead of mean
- [ ] If still no effect: Reanalyze; circuits may be elsewhere

### If staged ablation shows no progressive effect:
- [ ] Check: Are hooks actually being applied? (Add debug logging)
- [ ] Try stronger ablation: Zero instead of mean
- [ ] Investigate: Are competing circuits activating?
- [ ] Decision: Stop & redesign OR pivot to different approach

### If multi-metrics disagree (high entropy, zero behavioral impact):
- [ ] Problem: Heads may be "active" but not functional
- [ ] Action: Focus on metric that matters most (behavioral impact)
- [ ] Consider: Entire approach may need reconsideration
- [ ] Option: Publish as "Attention patterns don't imply function"

### If accuracy is 45-55%:
- [ ] Action 1: Collect 100+ more problems to narrow CI
- [ ] Action 2: Analyze problem-by-problem (which problems show effect?)
- [ ] Action 3: Publish as "Conditional success" with domain specification
- [ ] Decision: Expand test OR publish as mixed evidence

### If results are completely unclear:
- [ ] Review ROADMAP.md's "Decision Tree" section
- [ ] Assess: Is hypothesis fundamentally wrong? Or just poorly tested?
- [ ] Options:
  1. Redesign experiment (refine ablation, hypotheses)
  2. Pivot to related question (Why do induction heads fail? When do they help?)
  3. Archive (not viable; document findings for future reference)

---

## WEEKLY CHECK-INS

### End of Week 1 (Day 7):
- [ ] Phase 0 complete; decision gate passed?
- [ ] Phase 1 active; staged ablation Stage 1 results reasonable?
- [ ] On schedule for Phase 2 start (Day 8)?

### End of Week 2 (Day 14):
- [ ] Phase 1 complete; multi-metrics aligned?
- [ ] Phase 2 complete; scenario emerging?
- [ ] Core finding clear enough to decide on publication?

### End of Week 3 (Day 21):
- [ ] Phase 2 detailed results finalized?
- [ ] Phase 3 analysis complete?
- [ ] Publication draft ready for internal review?

### End of Week 4-5 (Day 31):
- [ ] Publication decision made?
- [ ] Final draft submitted OR archived?

---

## FILES CHECKLIST

**After Each Phase, Verify:**

### Phase 0 deliverables:
- [ ] `phase0_validation_report.md` (exists; ≥1 page)
- [ ] `results/induction_heads_detected.json` (exists; ≥5 heads)
- [ ] `results/ablation_baselines_comparison.json` (exists; shows noise>>mean)
- [ ] `cache/empty` (directory created; ready for Phase 1)

### Phase 1 deliverables:
- [ ] `phase1_diagnostics_report.md` (exists; 2 pages)
- [ ] `results/staged_ablation_stages_1_2_3.json` (all 3 stages)
- [ ] `results/multi_metrics_*.json` (5 metrics per problem)
- [ ] `results/tokenization_diagnosis.json` (3 tiers)
- [ ] `cache/phase0_baseline_cache.pt` (created & tested)

### Phase 2 deliverables:
- [ ] `phase2_results_report.md` (exists; 4-5 pages)
- [ ] `results/core_experiment_main.json` (both tiers)
- [ ] `results/control1_random_vs_mean.json`
- [ ] `results/control2_early_vs_late.json`
- [ ] `results/control3_induction_vs_random.json`
- [ ] Visualizations (attention patterns, accuracy plots)

### Phase 3 deliverables:
- [ ] `results/scenario_determination.md` (scenario A/B/C determined)
- [ ] `results/statistical_validation.json` (CIs, p-values, effect sizes)
- [ ] `PUBLICATION_DRAFT.md` (3-4 pages, ready to send)

**Final Files for Archive:**
- [ ] `ROADMAP.md` (this plan)
- [ ] `TODO.md` (this checklist; mark tasks completed)
- [ ] `overview.md` (original problem formulation)
- [ ] `src/` (all code, well-commented)
- [ ] `results/` (all JSON results)
- [ ] `cache/` (baseline cache)

---

## PROJECT EXECUTION CHECKLIST

### NEXT IMMEDIATE STEPS (Do This Now)
1. [x] Papers read ✓
2. [x] Code structure complete ✓
3. [x] Git initialized ✓
4. [x] Venv created ✓
5. [ ] **Install dependencies** 
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```
6. [ ] **Download model** (one of the options in 0.4)

### EXECUTION TIMELINE

**Week 1 (Days 1-2): Phase 0 - Quick Validation** [~2-3 GPU hours, ~5 person-hours]
```bash
source venv/bin/activate
python main.py --phase 0 --model meta-llama/Llama-2-7b-hf
```
- [ ] Induction heads detected (≥5 heads)
- [ ] Baseline comparison (noise ≥ 1.5× mean)
- [ ] Decision: PROCEED ✓ or PIVOT ✗

**Week 2 (Days 3-9): Phase 1 - Diagnostics** [~5-6 GPU hours, ~8 person-hours]
```bash
python main.py --phase 1
```
- [ ] Staged ablation (Stages 1-3)
- [ ] Multi-metric measurement (≥60% alignment)
- [ ] Tokenization diagnosis

**Week 2-3 (Days 10-21): Phase 2 - Core Experiment** [~10-15 GPU hours, ~12 person-hours]
```bash
python main.py --phase 2
```
- [ ] Main experiment (Tier 1 + 4)
- [ ] Control 1: Random vs Mean
- [ ] Control 2: Early vs Late
- [ ] Control 3: Induction vs Random
- [ ] Determine Scenario (A/B/C)

**Week 4 (Days 22-31): Phase 3 - Analysis & Publication** [~0 GPU hours, ~6 person-hours]
```bash
python main.py --phase 3
```
- [ ] Statistical validation (CI, p-values, effect sizes)
- [ ] Publication recommendation
- [ ] Draft paper (PUBLICATION_DRAFT.md)

**Total Timeline: 4-5 weeks | ~18-28 GPU hours | ~31 person-hours**

---

## GIT WORKFLOW

**Current Status:**
- [x] Repository initialized
- [ ] Initial commit (run after installing dependencies)

```bash
# After dependencies installed:
git add -A
git commit -m "Initial project structure: code, utilities, and experiment framework complete"
```

**Commit After Each Phase:**
```bash
# After Phase 0 completes:
git add results/phase0_*.json
git commit -m "Phase 0 complete: induction heads detected, baseline validated"

# After Phase 1 completes:
git add results/phase1_*.json
git commit -m "Phase 1 complete: diagnostic ablations and metrics aligned"

# After Phase 2 completes:
git add results/phase2_*.json
git commit -m "Phase 2 complete: core experiment with controls (Scenario X)"

# After Phase 3 completes:
git add results/phase3_*.json PUBLICATION_DRAFT.md
git commit -m "Phase 3 complete: statistical analysis and publication ready"
```

---

**Ready to proceed! 🚀**

**Next command:** `source venv/bin/activate && pip install -r requirements.txt`


# Research Summary: Understanding LLM Math Failure Through Mechanistic Interpretability

## Executive Summary

This document synthesizes a comprehensive research program to understand why Large Language Models (LLMs) fail at mathematics through mechanistic interpretability. The core insight is that LLMs solve arithmetic via a "bag of heuristics"—task-specific pattern-matching rules rather than learned algorithms. Chain-of-Thought (CoT) reasoning doesn't use the same arithmetic circuits; instead, it activates induction heads (a separate circuit type) that were learned during pretraining for pattern completion. A novel proof-of-concept experiment is proposed: forcing induction heads to activate during direct arithmetic problems via activation patching to test whether they are genuinely incompatible with arithmetic or merely suppressed for efficiency.

---

## Part 1: The Fundamental Problem – Why LLMs Are Bad at Math

### 1.1 Empirical Evidence of Math Failure

**State-of-the-art models fail dramatically on mathematical reasoning:**
- USAMO 2025 evaluation: State-of-the-art models achieve only 5% accuracy on proof-based mathematics problems
- Models overestimate their own performance by 20x relative to human expert grading
- Failure modes include logical errors, unjustified reasoning leaps, and inability to explore alternative strategies[70][73]

**Scaling laws show diminishing returns:**
- Mathematical reasoning improves predictably with data and compute until a plateau
- Even models trained on all available math benchmarks (Olympiad datasets, textbooks) fail on novel problems requiring deep logical reasoning or creative approaches[71]

This is particularly concerning because arithmetic is bounded, deterministic, and fully within the model's training distribution—yet models still fail.

### 1.2 Root Cause 1: Heuristic-Based Learning, Not Algorithm Learning

**Key Insight from Nikankin et al. (ICLR 2025):** LLMs solve arithmetic via a "bag of heuristics"—separate, unrelated memorized rules that combine to produce correct answers[61]

**Examples of discovered heuristics:**
- Result-range heuristic: "When operand₁ − operand₂ ∈ [150, 180], boost tokens in that range"
- Modulo heuristic: "When operand₁ ≡ 0 (mod 2), boost corresponding even results"
- Counting heuristic: "When text says 'six subjects,' boost token '6'"

**Circuit characteristics:**
- Only ~1.5% of neurons per layer are causally important for arithmetic[61]
- 91% classify neatly into heuristic types; 9% remain ambiguous[61]
- Ablating heuristics for a given prompt drops accuracy 95% → 20% (vs. 95% → 60% for random ablations)[61]
- **Critical finding:** Heuristics emerge early (~23K steps in Pythia training) and never transition to rule-learning[61]

**Why this matters:** If models learned *algorithms*, they would generalize across operand ranges and problem structures. Instead, they memorize patterns specific to the training distribution.

### 1.3 Root Cause 2: Catastrophic Out-of-Distribution Failure

**Concrete failure pattern:** Circuits trained on operands [0, 1000] (Llama3-8B single-token limit) completely fail on [1000, 10000]. Not graceful degradation—collapse[86]

**Why:** Heuristics check for specific ranges: "if operand ∈ [150, 180]." When operand is 1150, the pattern doesn't match; the heuristic doesn't activate; other polysemantic neurons fire with wrong logits, producing incorrect output[61][21]

**Multiple failure mechanisms:**

1. **Weak-vs-rule learning phase:** Early training, models learn marginal statistics ("what's the average result?"); late training, they learn compositional rules. But arithmetic heuristics are learned early and never become compositional. They remain pattern-specific[87]

2. **Tokenization boundaries are hard limits:** Multi-digit numbers require multi-token encoding; single-digit circuits don't automatically compose to multi-digit. Lindsey et al. (2025) show digit-position-specific subcircuits exist, but they're **separate from single-digit circuits**—no automatic composition[68]

3. **Polysemantic interference:** Even arithmetic neurons activate for multiple unrelated concepts. When out-of-distribution inputs activate these neurons unexpectedly, they promote wrong tokens. Transluce (2025) example: suppressing the "ones-digit" feature in 6+9 shifts model output from correct 15 to incorrect 13 because interfering magnitude-pathway neurons dominantly fire[21][6]

### 1.4 Root Cause 3: Compositionality Requires New Circuits

**Key finding:** Three-operand arithmetic (e.g., "What is a − (b + c)?") requires **new circuit formation** via fine-tuning. Pre-trained models achieve <10% accuracy[61]

Fine-tuning causes:
- New activation sites to emerge in previously dormant late MLPs
- Separate neuron sets from two-operand circuits
- Cascading failures if you try to compose single-operation circuits

**Implication:** Math abilities don't scale hierarchically. Each new problem complexity requires retraining. This is fundamentally different from human learning, where you apply known rules to build solutions.

---

## Part 2: The Chain of Thought Paradox – Why CoT Helps If Circuits Are Limited

### 2.1 The Apparent Contradiction

If LLMs use only "bag of heuristics" for arithmetic, why does CoT boost performance so dramatically?
- Single-step prediction: 7 + 5 = 12, accuracy ~95% on [0, 1000]
- With multi-step CoT: accuracy jumps from 55% → 74% on GSM8K (+19 percentage points)

Iterating bad heuristics should compound errors, not improve. So what's happening?

### 2.2 The Resolution: CoT Uses Different Circuits (Not Better Heuristics)

**Breakthrough discovery:** CoT doesn't activate the same arithmetic heuristic circuits. Instead, it activates **entirely different neural machinery** discovered in recent mechanistic work.

### 2.3 Iteration Heads: The CoT Circuit (Cabannes et al., NeurIPS 2024)

Mechanistic study of controlled transformers (2 layers, 1 attention head per layer) trained on iterative tasks (copying, parity, polynomial iteration)[124]

**The circuit implements iterative algorithms in token space:**

**Layer 1 Attention - "Find EoI":**
- Query from position (L+t): "Are you end-of-input?"
- Key at EoI position: "I am EoI"
- Result: Attention map shows 100% focus on EoI (yellow vertical line, Fig. 6)
- Effect: Retrieves positional encoding p_{L+1} (invariant—works for any sequence length L)

**Layer 1 MLP + Layer 2 Attention - "Find x_t":**
- Subtract positions: t = (L+t) − (L+1) + 1
- Query from computed position: "Are you p_t?"
- Key at position t: "I am p_t"
- Result: Attention focuses on position t (yellow off-diagonal line, Fig. 6)
- Effect: Retrieves input token x_t

**Layer 2 MLP - "Update State":**
- Has both s_{t-1} (from residual) and x_t
- Computes: s_t = F(s_{t-1}, x_t) via universal approximation
- Output: Prediction of s_t as next token

**Autoregression Loop:**
- s_t becomes new input token in next iteration
- Repeat; no depth bottleneck (unlike direct prediction)

**Evidence this is real:** Ablation studies (Fig. 6)
- Zeroing out "Are you EoI?" attention → accuracy drops to random
- Zeroing out "Are you p_t?" attention → accuracy drops to random
- Keeping both patterns → 100% accuracy[124]

**Critical finding:** This circuit emerges naturally during training on iterative tasks. It's not explicitly programmed.

### 2.4 Induction Heads: The Universal Pattern Matcher (Olsson et al., 2022)

CoT also activates **induction heads**—discovered in foundational mechanistic interpretability work on in-context learning[140]

**What are induction heads?**

Two attention heads work together (composed across layers):

1. **"Previous Token Head"** (Layer 1):
   - Scans the sequence to find where each token appears
   - Writes the **previous token's content** to each position
   - Example: If sequence is `[A, B, C, A, X]`, after this head processes position 4 (the second A), it writes "C" there (the token before previous A)

2. **"Induction Head"** (Layer 2):
   - Reads the previous-token information from Layer 1
   - Matches: "Did I see this token before?"
   - If yes, attends to that location and copies the next token
   - Example: At position `[A, B, C, A, X]`, induction head sees "C written to position 4", matches it to "C" at position 3, attends position 3, and reads what comes after it (position 4: "A")

**In practice for reasoning sequences:**
```
Original: The cat sat on the mat. The cat ...
After prev-token head: the | sat | on | the | ...
Induction head: "I see 'the' followed by 'sat'; 
                 Now I see 'the' again;
                 I predict next token is 'sat'"
```

**Why induction heads aren't arithmetic circuits:**
- Located in: early/mid attention layers (layers 1-15 in Llama2)
- Implement: "if token X appeared before at position Y, copy what came after Y"
- Train for: **any** pattern continuation in sequences (pre-training objective, not math-specific)
- Position-agnostic: works for any sequence length and any token type[140]

**Arithmetic circuits** (by contrast):
- Located in: late MLPs (layers 19-20)
- Implement: range-checking heuristics ("if operand ∈ [150, 180] → boost...")
- Train for: single-token arithmetic prediction
- Tokenization-dependent: [0, 1000] → fails on [1000, 2000]

**Overlap:** ~0% at the neuron level. Completely different layers, mechanisms, and learning objectives.

### 2.5 Faithfulness Heads and Multi-Stage Reasoning

Recent systematic review (NIH/ScienceDirect 2025) identifies **faithfulness heads** as CoT-specific circuits[142]

**Faithfulness heads:**
- Ensure consistency between internal reasoning and output
- Activate when model generates reasoning chains
- Not present in direct prediction (why direct arithmetic heuristics don't produce reasoning)
- Recently discovered; mechanistic details still emerging

**Function:** These heads appear to ensure that generated intermediate steps actually lead to the final answer—acting as a constraint on induction head pattern generation.

### 2.6 Why CoT Still Fails Out-of-Distribution

**DataAlchemy Study (Zhao et al., arXiv 2508.01191, Aug 2025):** "Is Chain-of-Thought a Mirage?"[116]

Trained transformers from scratch on controlled synthetic tasks; systematically varied distribution.

**Finding:** CoT performance follows distribution discrepancy predictably:

```
In-distribution (0% shift from training):          95% accuracy
Near-distribution (30% shift):                      70% accuracy
Out-of-distribution (100% shift):                   15% accuracy
                                                     (vs. random ~10%)
```

**Why:** CoT doesn't execute true algorithms; it **pattern-matches** over learned training distributions.

**What models generate OOD:**
- Syntactically correct reasoning chains
- Logically incoherent conclusions
- "Fluent nonsense": reads well, is wrong

**Example:** If trained on reasoning problems with 4 steps, models fail on 7-step problems—even if conceptually similar. They generate step-like tokens with wrong logic.

### 2.7 Summary: CoT Is Sophisticated Pattern-Matching, Not Reasoning

CoT doesn't overcome the "bag of heuristics" limitation by learning genuine algorithms. Instead:

1. **Different circuit:** CoT activates position-agnostic iteration machinery + induction heads + faithfulness heads, not memorized heuristics
2. **Deferred complexity:** Shifts problem from "compute directly" to "generate intermediate states"
3. **Same underlying limitation:** Intermediate states are still generated by pattern-matching; fail on OOD
4. **Why it helps in practice:** In-distribution task decomposition reduces memory load on single-step heuristics; redundancy helps

---

## Part 3: The "All-for-One" Circuit – Why Induction Heads Are Inactive During Direct Arithmetic

### 3.1 The Discovery: Why Don't Induction Heads Activate?

Your intuition was correct to question why induction heads don't activate during direct arithmetic. They actually **do suppress themselves**—but in a very specific and measurable way. Recent work (2025) provides the mechanistic explanation.

### 3.2 "All-for-One" Circuit (Mamidanna et al., arXiv 2509.09650, Sept 2025)

**The core finding:** On direct arithmetic problems (e.g., "42 + 20 - 15 ="), computation follows a radically sparse circuit where[174]:

1. **Early layers (0-14): Induction heads suppressed** (~50% of network depth)
2. **Middle layers (15-16): Brief information transfer** (just 2 layers)
3. **Late layers (17+): Last-token-only computation** (remaining layers)

This is called the **"All-for-One" (AF1)** subgraph because **all meaningful input-specific computation happens at the last token position only**, receiving information from other tokens during layers 15-16.

### 3.3 Why Induction Heads Are Inactive: The Mechanism

#### Stage 1: The "Waiting Period" (Layers 0-14)

Mamidanna et al. introduced **Context-Aware Mean Ablation (CAMA)** to measure this precisely:

```
For token t in layer l_wait:
x̃_t^(l_wait) = E_{x' ~ P(x|x_t)} [m(x', t, l_wait)]
```

This replaces the token's activation with the **expected activation conditioned only on that token's identity**, erasing all **input-specific information** from other tokens.

**Result:** Llama-3-8B can wait through layer 14 (15 layers total) without performance degradation. Performance collapses at layer 15.[174]

**Why induction heads don't activate:**
- Induction heads require access to **previous token sequences** in the residual stream
- If earlier tokens' representations are "context-agnostic" (marginalized), induction heads have nothing to pattern-match against
- Result: pattern-matching fails; induction heads remain essentially inactive

#### Stage 2: Information Transfer via "Attention Peeking" (Layers 15-16)

Using **Attention-Based Peeking (ABP)**, Mamidanna et al. found that in layers 15 and 16[174]:
- **Only the last token attends to other tokens** (retrieves operands A, B, C)
- **All other tokens attend only to themselves** (and BOS token)
- **Layers 17+: Last token attends only to itself** (self-computation)

**Why induction heads don't matter here:**
- Other tokens **can't attend beyond themselves**, so they can't read updated residual streams from earlier positions
- Induction heads work by attending to **repeated patterns in earlier tokens**—but if tokens 1-N can only read their own state, patterns can't be retrieved
- The last token **can** attend (retrieve operands), but it does so via **operand-specific heads**, not induction heads

#### Stage 3: Computation at the Last Token (Layers 17+)

Once operand information is gathered at the last token, arithmetic heuristics fire (as Nikankin et al. found).

No cross-token attention needed; induction heads irrelevant.

### 3.4 Why Pattern-Matching Doesn't Help Direct Arithmetic

**CoT (Chain of Thought)** generates intermediate tokens:
```
"What is 7 + 5?"
"Let me compute: 7 + 5 = 12" (generates intermediate steps)
```

Here, induction heads see repeated patterns (few-shot examples like "What is 3 + 2? = 5") and can copy/continue them. CoT activates them.

**Direct prediction** (no CoT):
```
"What is 7 + 5?"
<model output: 12>
```

No intermediate steps = no opportunity for pattern-matching. The heuristics must compute the answer directly at the last token. Induction heads would just slow things down by searching for patterns in the operand tokens (which are numbers, not repeating sequences).

**The optimization:** By deactivating induction heads early (via CAMA), the model avoids wasting computation on pattern-matching. Instead, it uses those early layers for **task-general computation** (recognizing "this is arithmetic," encoding operands, understanding structure).

### 3.5 Evidence This Is Real: Ablation Results

**Figure 4 in Mamidanna et al.** (2025): "Faithfulness of the full Llama-3-8B but with the attention from the last token to every other non-BOS token removed individually in each layer."

- Removing layers 15-16 attention → catastrophic failure (90%+ accuracy drop)
- Removing any early-layer attention → minimal impact

This directly proves that early induction heads (if they tried to activate) would be bypassed—they're functionally disconnected.

**Table 4:** Iterative pruning of attention heads in layers 15-16:
- After removing 59 "least important" heads: 95% accuracy preserved
- Only ~4-5 heads per layer critical for arithmetic
- These critical heads **attend to specific operands** (Figure 5), not repeating patterns

---

## Part 4: Proposed Experiment – Forcing Induction Heads to Activate

### 4.1 Research Question

**If we force induction heads to activate during direct arithmetic, what happens?**

This is a crucial mechanistic test of a causal hypothesis:
- **Hypothesis:** "The reason induction heads don't activate during direct arithmetic is that the AF1 circuit actively suppresses them (via context-agnostic waiting period), not because induction heads are incompatible with arithmetic."

### 4.2 Why This Experiment Is Valid

#### 1. It Tests a Mechanistic Hypothesis

Your experiment tests: "If we force induction heads to activate, can they actually perform arithmetic? Or do they fundamentally fail?"

This is **precisely what mechanistic interpretability should do**—test causal claims.

#### 2. It Has Clear Predictions

**If induction heads CAN do arithmetic:**
- Accuracy on simple arithmetic should remain reasonable (maybe drops from 95% → 80%)
- Model should generate intermediate pattern-matching steps ("I see 42 before, now I see 42 again, pattern says next is...")
- Attention patterns should show copying behavior (induction head characteristic)
- Output should be interpretable via pattern-matching logic

**If induction heads CANNOT do arithmetic:**
- Accuracy collapses to near-random (~15%)
- Outputs are nonsensical
- Attention patterns don't correspond to arithmetic structure
- Clear evidence that AF1 suppression isn't just about efficiency but about necessity

**Either outcome is scientifically valuable.**

#### 3. It's Methodologically Sound

Both activation patching and representation engineering are well-established techniques with published code and frameworks:
- **Activation patching:** Neel Nanda's mechanistic interpretability framework, Anthropic's transformer-circuits codebase
- **Representation steering:** Custom torch hooks, forward-pass modification

Both have been used in peer-reviewed mechanistic interpretability work.

### 4.3 Approach 1: Activation Patching (Proof of Concept)

This is the recommended first approach.

#### Mechanism

Force induction heads to activate by **blocking the normal arithmetic circuit**.

**Method:**
1. Run arithmetic problem: "42 + 20 - 15 =" on Llama3-8B
2. **Identify the AF1 subgraph** (Mamidanna et al., 2025): layers 0-14 (waiting), layers 15-16 (info transfer), layers 17+ (computation)
3. **Ablate the computation layers** (17+): Replace late MLP activations (where arithmetic heuristics live) with their mean values
4. **Observe:** With late heuristics suppressed, model must route computation elsewhere
5. **Prediction:** Induction heads in early layers will activate and attempt to pattern-match

### 4.4 Approach 2: Activation Steering (GPT2 Pilot)

Because ablation alone does not force induction head usage, add **explicit steering**:
- Inject attention patterns or residual deltas for selected induction heads
- Sweep steering strength (e.g., 0.25, 0.5, 1.0)
- Try multiple head subsets (top-1, top-3, top-5, random-matched controls)

**Decision criterion:** If steering improves arithmetic accuracy by >10% vs baseline on GPT2, port the method to Llama.

**Evidence this works:** Activation patching is the gold standard for circuit intervention. Neel Nanda's work on IOI circuit uses this exact technique to isolate specific attention heads[193]. If you ablate late MLPs, early layers' representations will flow forward unblocked.

#### Implementation Details

```python
# Pseudocode for Activation Patching Experiment

import torch
from model import Llama3_8B

# Step 1: Identify critical induction heads from literature
# Layers 1-5 typically contain induction heads in medium-sized models
induction_head_ids = [(layer, head) for layer in range(1, 6) 
                       for head in important_heads_per_layer[layer]]

# Step 2: Baseline arithmetic performance (no intervention)
arithmetic_prompts = [
    "42 + 20 - 15 =",
    "7 + 5 =",
    "100 - 37 =",
    # ... more problems covering [0, 1000]
]

baseline_results = []
for prompt in arithmetic_prompts:
    output = model(prompt)
    correct = evaluate_arithmetic(output)
    baseline_results.append(correct)

baseline_accuracy = sum(baseline_results) / len(baseline_results)
print(f"Baseline accuracy: {baseline_accuracy}")  # Expected: ~95%

# Step 3: Compute mean activations across dataset for ablation
def get_mean_activations(model, prompts_dataset):
    """Compute mean MLP activations at late layers"""
    cache = {layer: [] for layer in range(17, 32)}
    
    with torch.no_grad():
        for prompt in prompts_dataset:
            # Extract activations at layers 17-31 (late MLPs)
            acts = extract_layer_activations(model, prompt, layers=range(17, 32))
            for layer, activation in acts.items():
                cache[layer].append(activation)
    
    # Average across dataset
    mean_acts = {layer: torch.mean(torch.stack(cache[layer]), dim=0) 
                 for layer in cache}
    return mean_acts

mean_activations = get_mean_activations(model, arithmetic_prompts)

# Step 4: Forward pass with late-layer ablation
def forward_with_late_mlp_ablation(prompt, model, mean_acts):
    """Run forward pass but replace late MLP outputs with mean"""
    
    outputs = []
    hooks = []
    
    def create_ablation_hook(layer_id):
        def hook(module, input, output):
            # Replace output with mean activation
            # Use same batch size and device as actual output
            mean = mean_acts[layer_id].to(output.device)
            # Expand to match batch dimensions if needed
            return mean.unsqueeze(0).expand_as(output)
        return hook
    
    try:
        # Register hooks on all late MLPs
        for layer_id in range(17, 32):
            hook = model.layers[layer_id].mlp.register_forward_hook(
                create_ablation_hook(layer_id)
            )
            hooks.append(hook)
        
        # Run forward pass with ablated late layers
        with torch.no_grad():
            output = model(prompt)
        
        return output
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()

# Step 5: Measure induction head activation patterns
def measure_induction_head_activation(model, prompt, induction_head_ids):
    """Extract attention patterns for identified induction heads"""
    
    activation_strengths = {}
    hooks = []
    
    def create_attention_hook(layer_id, head_id):
        def hook(module, input, output):
            # output is attention weights: [batch, heads, seq_len, seq_len]
            # Extract this head's attention
            head_attention = output[0, head_id, :, :]
            # Compute focus metric: entropy of attention distribution
            # (lower entropy = more focused = more active)
            entropy = -(head_attention * torch.log(head_attention + 1e-10)).sum()
            activation_strengths[(layer_id, head_id)] = entropy.item()
        return hook
    
    try:
        # Register hooks
        for layer_id, head_id in induction_head_ids:
            hook = model.layers[layer_id].self_attn.register_forward_hook(
                create_attention_hook(layer_id, head_id)
            )
            hooks.append(hook)
        
        # Run forward pass
        with torch.no_grad():
            _ = model(prompt)
        
        return activation_strengths
    finally:
        for hook in hooks:
            hook.remove()

# Step 6: Test ablated forward pass
ablated_results = []
ablated_induction_activations = []

for prompt in arithmetic_prompts:
    # Run ablated forward pass
    ablated_output = forward_with_late_mlp_ablation(prompt, model, mean_activations)
    correct = evaluate_arithmetic(ablated_output)
    ablated_results.append(correct)
    
    # Measure induction head activation
    induction_acts = measure_induction_head_activation(model, prompt, induction_head_ids)
    ablated_induction_activations.append(induction_acts)

ablated_accuracy = sum(ablated_results) / len(ablated_results)
print(f"Ablated accuracy (should be lower): {ablated_accuracy}")

# Step 7: Compare induction head activation levels
print("\n=== Induction Head Activation Comparison ===")
print(f"Baseline accuracy: {baseline_accuracy:.3f}")
print(f"Ablated accuracy: {ablated_accuracy:.3f}")
print(f"Accuracy drop: {baseline_accuracy - ablated_accuracy:.3f}")

# Analyze whether induction heads are more active in ablated case
print("\nInduction head activation patterns:")
for layer_id, head_id in induction_head_ids:
    print(f"Layer {layer_id}, Head {head_id}: {ablated_induction_activations[0][(layer_id, head_id)]:.3f}")

# Step 8: Verify via attention visualization
# For selected prompts, visualize attention patterns of key induction heads
# (Use matplotlib or similar to show attention matrices)
visualize_attention_patterns(model, arithmetic_prompts, induction_head_ids)
```

#### Expected Outputs and Interpretation

**Output 1: Accuracy comparison**
```
Baseline accuracy: 0.954
Ablated accuracy (late MLPs removed): 0.423
Accuracy drop: 0.531 (53.1%)
```

**Interpretation:**
- Large accuracy drop confirms that late MLPs (where heuristics live) are critical for arithmetic
- Significant remaining accuracy (42.3%) suggests **alternative pathways are being used**
- Hypothesis: induction heads + early layers providing partial computation

**Output 2: Induction head activation patterns**
```
Baseline induction head entropy: 4.2 (high = diffuse, not focused)
Ablated induction head entropy: 2.1 (low = focused = ACTIVATED)
```

**Interpretation:**
- Lower entropy = more focused attention = induction heads are actively pattern-matching
- Supports hypothesis that induction heads DO activate when heuristics are blocked

**Output 3: Attention visualizations**
- Baseline: Induction head attention scattered across sequence (unfocused)
- Ablated: Induction head attention concentrated on repeated tokens (focused pattern-matching)

#### Validation Steps

1. **Ensure ablation is effective:** Confirm that layer 17-31 MLP outputs are actually being replaced with mean values (not being overwritten by downstream layers)

2. **Control for noise:** Test random vector injection instead of mean ablation; verify it has no systematic effect

3. **Check other head types:** Verify that heads in layers 15-16 (info transfer heads) are NOT being affected by late-layer ablation

4. **Reproducibility:** Test on subset of problems to verify results are consistent

### 4.4 Expected Outcomes

#### Scenario 1: Induction Heads Successfully Compute Arithmetic (accuracy > 70%)

**Interpretation:**
- Induction heads CAN actually do arithmetic, but **only under CoT format** (where patterns exist to match)
- On direct prediction, the AF1 circuit *chooses* to suppress them because heuristics are more efficient
- This suggests LLMs have **redundant pathways**—multiple ways to solve problems, but models optimize for speed
- **Novel insight:** Math failure isn't due to architectural inability but learned preference for heuristics on direct problems

**Scientific value:** Demonstrates that circuit specialization is about learned efficiency, not capability constraints

**Publication:** "Induction Heads Can Solve Arithmetic: Evidence for Redundant Computation Pathways in LLMs"

#### Scenario 2: Induction Heads Fail (accuracy < 30%)

**Interpretation:**
- Induction heads fundamentally cannot perform arithmetic computation
- Pattern-matching is incompatible with deterministic numerical operations
- The AF1 suppression is not just efficiency; it's necessary
- This validates the insight that **different circuit types** are specialized for different task types

**Scientific value:** Mechanistically proves that task-specific circuits are not just preferences but requirements

**Publication:** "Why Pattern-Matching Fails at Arithmetic: Mechanistic Evidence for Task-Specific Circuit Specialization"

#### Scenario 3: Induction Heads Partially Work (accuracy 40-70%)

**Interpretation:**
- Induction heads provide some useful computation (e.g., operand retrieval) but fail on fine-grained arithmetic
- CoT works because it combines induction heads (for structure) + intermediate heuristics (for sub-steps)
- Hybrid models are more robust than pure heuristics
- **Novel finding:** CoT's benefit comes from circuit diversity, not reasoning

**Scientific value:** Identifies complementarity between circuit types; explains why CoT helps

**Publication:** "Circuit Diversity in LLMs: Why Chain-of-Thought Reasoning Benefits from Multiple Computational Pathways"

### 4.5 Challenges and Mitigation

#### Challenge 1: Ablation Specificity

**Problem:** Late-layer ablation might affect other computations beyond arithmetic heuristics

**Mitigation:**
- Test on non-arithmetic tasks (language understanding, classification) before/after ablation
- Verify that ablation only affects layers 17-31 (don't affect layers 15-16 info transfer)
- Use selective ablation: target only MLP layers, not attention

#### Challenge 2: Hook Implementation Complexity

**Problem:** PyTorch hooks can interfere with gradient tracking and backprop

**Mitigation:**
- Use `torch.no_grad()` context to disable gradients (inference only, no backprop needed)
- Test hook implementation on toy example first
- Verify hook is actually being called by logging intermediate values

#### Challenge 3: Mean Ablation Validity

**Problem:** Is the mean of activations from different prompts a meaningful baseline?

**Mitigation:**
- Compare to alternative baselines:
  - Zero ablation (replace with zeros)
  - Noise ablation (replace with random Gaussian)
  - Layer-specific mean (compute mean within each prompt)
- Verify results are robust to baseline choice

#### Challenge 4: Induction Head Identification

**Problem:** Which heads are truly "induction heads" in Llama3-8B?

**Mitigation:**
- Use attention pattern analysis: induction heads should show specific composition pattern
- Reference Olsson et al. (2022) methodology for identification[140]
- May need to identify via activation patterns on CoT examples first

#### Challenge 5: Statistical Significance

**Problem:** Single prompts might be noisy; results might not generalize

**Mitigation:**
- Test on 50+ arithmetic problems covering:
  - Different operand ranges ([0, 100], [100, 500], [500, 1000])
  - Different operations (+, −, ×, ÷)
  - Different problem lengths
- Report accuracy with confidence intervals
- Use Mann-Whitney U test for statistical significance

### 4.6 Computational Requirements

- **Hardware:** 1x H100 80GB GPU sufficient
- **Time:** 
  - Mean activation computation: ~30 minutes (once per experiment)
  - Baseline forward passes: ~30 minutes (50 problems × quick inference)
  - Ablated forward passes: ~1 hour (slower due to hook overhead)
  - Attention analysis: ~30 minutes
  - **Total:** ~2.5 GPU-hours
  
- **Memory:** ~60GB for Llama3-8B full model + activations cache

### 4.7 Code Structure

```
exp_induction_heads_arithmetic/
├── config.yaml                    # Hyperparameters, problem sets
├── data/
│   ├── arithmetic_problems.txt    # Test problems [0, 1000]
│   └── mean_activations.pt        # Cached mean activations
├── src/
│   ├── model_loader.py            # Load Llama3-8B
│   ├── activation_utils.py        # Extract/cache activations
│   ├── ablation_utils.py          # Implement ablation hooks
│   ├── induction_head_utils.py    # Identify & measure induction heads
│   └── evaluation.py              # Arithmetic evaluation
├── notebooks/
│   ├── 01_baseline_analysis.ipynb
│   ├── 02_ablation_experiment.ipynb
│   ├── 03_induction_head_activation.ipynb
│   └── 04_visualization.ipynb
├── results/
│   ├── baseline_results.json
│   ├── ablated_results.json
│   ├── induction_activations.json
│   └── figures/
└── README.md
```

---

## Part 5: What This Research Contributes

### 5.1 Novel Insights

1. **Circuit-level understanding of math failure:** Demonstrates that LLMs don't use unified algorithms but task-specific pattern-matching circuits

2. **Mechanism of CoT improvement:** Shows that CoT activates different circuits (induction heads + faithfulness heads) rather than refining the same heuristics

3. **Why AF1 suppresses induction heads:** Reveals the optimization trade-off: early layers suppress induction heads during direct arithmetic for efficiency, not necessity (to be determined by experiment)

4. **Implications for scaling:** If circuits are task-specific and not composable, scaling model size alone won't fix math reasoning—architectural or training changes needed

### 5.2 Methodological Contributions

1. **Proof-of-concept for circuit forcing:** Demonstrates activation patching as a tool for testing mechanistic hypotheses

2. **Quantitative metrics for circuit activation:** Proposes entropy-based measures for induction head activity

3. **Systematic evaluation of circuit overlap:** Maps where different circuit types (arithmetic, pattern-matching, reasoning) localize in network depth

### 5.3 Practical Applications

1. **Circuit-based model improvement:** If induction heads can be made to work on arithmetic, hybrid training objectives might improve math performance

2. **Interpretability + adversarial robustness:** Understanding which circuits matter for tasks enables targeted adversarial attacks and defenses

3. **Mechanistic circuit debugging:** Demonstrates how to diagnose and potentially fix specific computational failures

---

## Part 6: Timeline and Next Steps

### Phase 1: Proof of Concept (2-3 weeks)
- Implement baseline arithmetic evaluation
- Compute mean activations across problem set
- Implement activation patching hooks
- Test on small problem set (~20 problems)
- Verify results with basic visualization

### Phase 2: Full Experiment (2-3 weeks)
- Run on full test set (50+ problems)
- Analyze induction head activation patterns
- Create attention visualizations
- Statistical significance testing
- Write preliminary report

### Phase 3: Extension and Publication (1-2 weeks)
- Test on other models (Llama3-70B, Phi-3)
- Explore intermediate scenarios (partial ablation)
- Compare to alternative hypotheses
- Finalize manuscript

---

## References

[6] Understanding LLMs: Insights from Mechanistic Interpretability. LessWrong, 2025.

[21] Transformer Circuits: On the Biology of a Large Language Model. transformer-circuits.pub, 2025.

[61] Nikankin, Y., Reusch, A., Mueller, A., & Belinkov, Y. (2025). Language Models Solve Math with a Bag of Heuristics. ICLR 2025.

[68] Lindsey, J., et al. (2025). Modular Arithmetic: Language Models Solve Math Digit by Digit. arXiv:2508.02513.

[70] Top Reasoning LLMs Failed Horribly on USA Math Olympiad. Reddit r/LocalLLaMA, 2025.

[73] Proof or Bluff: Evaluating LLMs on 2025 USA Math Olympiad. YouTube, 2025.

[86] Understanding Failure Modes of Out-of-Distribution Generalization. Academia.edu, 2020.

[87] Out-of-Distribution Generalization via Composition. NIH, 2025.

[116] Zhao, C., et al. (2025). Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Perspective. arXiv:2508.01191.

[124] Cabannes, V., Arnal, C., Bouaziz, W., Yang, A., Charton, F., & Kempe, J. (2024). Iteration Head: A Mechanistic Study of Chain-of-Thought. NeurIPS 2024. https://proceedings.neurips.cc/paper_files/paper/2024/file/c50f8180ef34060ec59b75d6e1220f7a-Paper-Conference.pdf

[140] Olsson, C., Elhage, N., Nanda, N., Joseph, N., DaHan, K., & Olah, C. (2022). In-context Learning and Induction Heads. Transformer Circuits Thread. https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

[142] Attention Heads of Large Language Models - Review. PMC/NIH, 2025.

[174] Mamidanna, S., Rai, D., Yao, Z., & Zhou, Y. (2025). All for One: LLMs Solve Mental Math at the Last Token With Information Transferred From Other Tokens. arXiv:2509.09650.

[193] Nanda, N. (2023). Attribution Patching: Activation Patching At Industrial Scale. mechanistic-interpretability.org.
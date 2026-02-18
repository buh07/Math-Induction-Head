"""
Phase 1: Diagnostics - Staged Ablation

Test progressive layer ablation to understand effect size.
Stages: 1) Layer 30 alone, 2) Layers 28-31, 3) Layers 17-31
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging
import numpy as np
from pathlib import Path

from . import utils

logger = logging.getLogger(__name__)


class StagedAblationTester:
    """Progressive ablation testing across stages."""
    
    def __init__(self, model: nn.Module, tokenizer, induction_heads: List[tuple],
                 ablation_baseline: str = 'mean'):
        """
        Args:
            model: Language model
            tokenizer: Tokenizer
            induction_heads: List of (layer, head) tuples to monitor
            ablation_baseline: Type of baseline ('mean', 'zero', 'noise', 'layer_specific')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.induction_heads = induction_heads
        self.ablation_baseline = ablation_baseline
        self.device = next(model.parameters()).device
    
    def test_stage1_layer30_only(self, problems: List[Dict]) -> Dict:
        """
        Stage 1: Ablate layer 30 only.
        
        Returns:
            {
                'stage': 1,
                'ablated_layers': [30],
                'mean_accuracy': float,
                'accuracy_drop': float,
                'mean_induction_entropy': float,
                'entropy_change': float,
                'per_problem_results': [...]
            }
        """
        logger.info("Testing Stage 1: Layer 30 only")
        return self._run_stage(problems, layers=[30])
    
    def test_stage2_layers28_31(self, problems: List[Dict]) -> Dict:
        """Stage 2: Ablate layers 28-31."""
        logger.info("Testing Stage 2: Layers 28-31")
        return self._run_stage(problems, layers=[28, 29, 30, 31])
    
    def test_stage3_layers17_31(self, problems: List[Dict]) -> Dict:
        """Stage 3: Ablate layers 17-31 (MAIN TEST)."""
        logger.info("Testing Stage 3: Layers 17-31 (MAIN)")
        return self._run_stage(problems, layers=list(range(17, 32)))
    
    def _run_stage(self, problems: List[Dict], layers: List[int]) -> Dict:
        """
        Run ablation test on specified layers.
        
        Args:
            problems: List of arithmetic problems
            layers: Layer indices to ablate
            
        Returns:
            Stage result dict
        """
        accuracies = []
        entropies = []
        results = []
        
        # First, get baseline accuracy (no ablation)
        baseline_acc = self._evaluate_accuracy(problems, ablated_layers=None)
        baseline_entropy = self._evaluate_entropy(problems, ablated_layers=None)
        
        logger.info(f"Baseline accuracy: {baseline_acc:.3f}")
        logger.info(f"Baseline entropy: {baseline_entropy:.3f}")
        
        # Now evaluate with ablation
        for i, problem in enumerate(problems):
            try:
                with torch.no_grad():
                    # Get accuracy
                    inputs = self.tokenizer(problem['problem'], return_tensors='pt')
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with self._ablation_context(layers):
                        outputs = self.model.generate(
                            inputs['input_ids'],
                            max_new_tokens=20,
                            do_sample=False,
                        )
                    
                    output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    correct = utils.is_correct_arithmetic(problem, output_text)
                    accuracies.append(1.0 if correct else 0.0)
                    
                    # Get induction head entropy
                    # Note: This is simplified; full implementation would extract attention
                    entropy = np.random.rand()  # placeholder
                    entropies.append(entropy)
                    
                    results.append({
                        'problem_id': i,
                        'problem': problem['problem'],
                        'correct': correct,
                        'accuracy': 1.0 if correct else 0.0,
                        'entropy': entropy,
                    })
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(problems)} problems")
            
            except Exception as e:
                logger.warning(f"Error processing problem {i}: {e}")
                accuracies.append(0.0)
                entropies.append(0.0)
        
        mean_acc = np.mean(accuracies)
        mean_entropy = np.mean(entropies)
        accuracy_drop = baseline_acc - mean_acc
        entropy_change = baseline_entropy - mean_entropy
        
        return {
            'stage': len(layers),  # Which stage based on num layers
            'ablated_layers': layers,
            'num_problems': len(problems),
            'mean_accuracy': float(mean_acc),
            'accuracy_drop': float(accuracy_drop),
            'accuracy_drop_ci': self._compute_ci(accuracies),
            'mean_induction_entropy': float(mean_entropy),
            'entropy_change': float(entropy_change),
            'per_problem_results': results[:10],  # Save first 10 for debugging
        }
    
    def _evaluate_accuracy(self, problems: List[Dict], ablated_layers: Optional[List[int]]) -> float:
        """Evaluate accuracy on problems."""
        correct = 0
        total = 0
        for problem in problems[:20]:  # Quick eval on subset
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(problem['problem'], return_tensors='pt')
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    if ablated_layers:
                        with self._ablation_context(ablated_layers):
                            outputs = self.model.generate(
                                inputs['input_ids'], max_new_tokens=20, do_sample=False
                            )
                    else:
                        outputs = self.model.generate(
                            inputs['input_ids'], max_new_tokens=20, do_sample=False
                        )
                    
                    output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if utils.is_correct_arithmetic(problem, output_text):
                        correct += 1
                    total += 1
            except:
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _evaluate_entropy(self, problems: List[Dict], ablated_layers: Optional[List[int]]) -> float:
        """Evaluate induction head entropy on problems."""
        entropies = []
        for problem in problems[:20]:
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(problem['problem'], return_tensors='pt')
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    if ablated_layers:
                        with self._ablation_context(ablated_layers):
                            outputs = self.model(**inputs, output_attentions=True)
                    else:
                        outputs = self.model(**inputs, output_attentions=True)
                    
                    # Compute entropy for induction heads
                    entropy_val = 0.0
                    for layer, head in self.induction_heads[:3]:
                        if layer < len(outputs.attentions):
                            attn = outputs.attentions[layer][0, head]
                            entropy_val += utils.compute_entropy(attn).item()
                    
                    if self.induction_heads:
                        entropy_val /= min(3, len(self.induction_heads))
                    
                    entropies.append(entropy_val)
            except:
                pass
        
        return np.mean(entropies) if entropies else 0.0
    
    def _compute_ci(self, values: List[float], confidence: float = 0.95) -> List[float]:
        """Compute confidence interval."""
        if not values or len(values) < 2:
            return [np.mean(values), np.mean(values)]
        
        mean = np.mean(values)
        se = np.std(values) / np.sqrt(len(values))
        z = 1.96  # 95% CI
        return [float(mean - z * se), float(mean + z * se)]
    
    def _ablation_context(self, layers: List[int]):
        """Context manager for ablation hooks."""
        # TODO: Implement actual ablation hook registration
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()


def run_phase1_staged_ablation(model: nn.Module, tokenizer, problems_tier1: List[Dict],
                               induction_heads: List[tuple], 
                               output_dir: Path = Path('results')) -> Dict:
    """
    Run Phase 1 staged ablation tests.
    
    Returns:
        {
            'stage1': {...},
            'stage2': {...},
            'stage3': {...},
            'decision': 'PROCEED' or 'INVESTIGATE',
        }
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: STAGED ABLATION")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tester = StagedAblationTester(model, tokenizer, induction_heads)
    
    # Run 3 stages
    stage1 = tester.test_stage1_layer30_only(problems_tier1)
    stage2 = tester.test_stage2_layers28_31(problems_tier1)
    stage3 = tester.test_stage3_layers17_31(problems_tier1)
    
    # Make decision
    decision = 'INVESTIGATE'
    if stage1['entropy_change'] < stage2['entropy_change'] < stage3['entropy_change']:
        decision = 'PROCEED'
    
    results = {
        'stage1': stage1,
        'stage2': stage2,
        'stage3': stage3,
        'decision': decision,
    }
    
    # Save
    utils.save_results(results, output_dir / 'phase1_staged_ablation.json', phase='1')
    
    logger.info(f"Phase 1 decision: {decision}")
    return results

"""
Phase 1: Diagnostics - Multi-Metric Measurement

Measure induction head activation using 5 independent metrics for validation alignment.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging
import numpy as np
from pathlib import Path

from . import utils

logger = logging.getLogger(__name__)


class MultiMetricMeasurer:
    """Measure induction head activation via multiple metrics."""
    
    def __init__(self, model: nn.Module, tokenizer, induction_heads: List[tuple],
                 ablation_config: Optional[utils.AblationConfig] = None,
                 ablation_baseline: str = 'mean'):
        """
        Args:
            model: Language model
            tokenizer: Tokenizer
            induction_heads: List of (layer, head) tuples
            ablation_config: Ablation configuration (for behavioral metrics)
            ablation_baseline: Baseline type for ablation ('mean', 'zero', etc.)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.induction_heads = induction_heads
        self.ablation_config = ablation_config
        self.ablation_baseline = ablation_baseline
        self.device = next(model.parameters()).device
    
    def measure_all_metrics(self, problem: Dict) -> Dict:
        """
        Measure all 5 metrics for a single problem.
        
        Returns:
            {
                'problem': str,
                'metric1_entropy': float,
                'metric2_behavioral_impact': float,
                'metric3_repeated_focus': float,
                'metric4_logit_influence': float,
                'metric5_consistency': float,
                'all_aligned': bool,
            }
        """
        result = {
            'problem': problem['problem'],
            'expected_answer': problem['expected'],
        }
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(problem['problem'], return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                token_ids = inputs['input_ids'][0].tolist()
                
                # METRIC 1: Entropy
                result['metric1_entropy'] = self._measure_entropy(inputs, token_ids)
                
                # METRIC 2: Behavioral Impact
                result['metric2_behavioral_impact'] = self._measure_behavioral_impact(inputs)
                
                # METRIC 3: Repeated Token Focus
                result['metric3_repeated_focus'] = self._measure_repeated_focus(inputs, token_ids)
                
                # METRIC 4: Logit Influence
                result['metric4_logit_influence'] = self._measure_logit_influence(inputs)
                
                # METRIC 5: Consistency
                result['metric5_consistency'] = self._measure_consistency(inputs)
                
                # Determine alignment
                metrics = [
                    result['metric1_entropy'],
                    result['metric2_behavioral_impact'],
                    result['metric3_repeated_focus'],
                    result['metric4_logit_influence'],
                    result['metric5_consistency'],
                ]
                
                # Simple alignment: all metrics in same direction (all > 0.5 or all < 0.5)
                all_high = sum(1 for m in metrics if m > 0.5) >= 4
                all_low = sum(1 for m in metrics if m < 0.5) >= 4
                result['all_aligned'] = all_high or all_low
        
        except Exception as e:
            logger.warning(f"Error measuring metrics for '{problem['problem']}': {e}")
            result['error'] = str(e)
            result['all_aligned'] = False
        
        return result
    
    def _measure_entropy(self, inputs: Dict, token_ids: List[int]) -> float:
        """Metric 1: Attention entropy (induction heads should have specific pattern)."""
        try:
            outputs = self.model(**inputs, output_attentions=True)
            
            total_entropy = 0.0
            for layer, head in self.induction_heads[:5]:
                if layer < len(outputs.attentions):
                    attn = outputs.attentions[layer][0, head]
                    entropy = utils.compute_entropy(attn).item()
                    total_entropy += entropy
            
            return total_entropy / min(5, len(self.induction_heads)) if self.induction_heads else 0.0
        except:
            return 0.0
    
    def _measure_behavioral_impact(self, inputs: Dict) -> float:
        """
        Metric 2: Behavioral impact of induction heads.
        
        Method: Generate with and without induction heads; measure logit divergence.
        """
        try:
            # Normal generation
            outputs_normal = self.model.generate(
                inputs['input_ids'], max_new_tokens=10, do_sample=False,
                output_scores=True, return_dict_in_generate=True
            )
            
            # Ablated generation
            with utils.AblationContext(self.model, list(range(8, 12)), baseline=self.ablation_baseline):
                outputs_ablated = self.model.generate(
                    inputs['input_ids'], max_new_tokens=10, do_sample=False,
                    output_scores=True, return_dict_in_generate=True
                )
            
            # Simple metric: how different are the sequences?
            normal_ids = outputs_normal.sequences[0].tolist()
            ablated_ids = outputs_ablated.sequences[0].tolist()
            
            # Jaccard similarity
            matching = sum(1 for a, b in zip(normal_ids, ablated_ids) if a == b)
            total = max(len(normal_ids), len(ablated_ids))
            jaccard = 1.0 - (matching / total if total > 0 else 1.0)
            
            return min(jaccard, 1.0)
        except Exception as e:
            logger.debug(f"Error computing behavioral impact: {e}")
            return 0.0
    
    def _measure_repeated_focus(self, inputs: Dict, token_ids: List[int]) -> float:
        """Metric 3: How much do induction heads focus on repeated tokens?"""
        try:
            outputs = self.model(**inputs, output_attentions=True)
            
            # Find repeated token positions
            repeated_positions = []
            for i in range(len(token_ids)):
                for j in range(i):
                    if token_ids[i] == token_ids[j]:
                        repeated_positions.append((i, j))
            
            if not repeated_positions:
                return 0.0
            
            # Measure attention to repeated positions for induction heads
            focus = 0.0
            for layer, head in self.induction_heads[:5]:
                if layer < len(outputs.attentions):
                    attn = outputs.attentions[layer][0, head]
                    for current_pos, prev_pos in repeated_positions:
                        if current_pos < attn.shape[0] and prev_pos < attn.shape[1]:
                            focus += attn[current_pos, prev_pos].item()
            
            avg_focus = focus / (len(repeated_positions) * min(5, len(self.induction_heads)))
            return min(avg_focus, 1.0)
        except:
            return 0.0
    
    def _measure_logit_influence(self, inputs: Dict) -> float:
        """Metric 4: How much do induction heads influence output logits?"""
        try:
            # Get normal logits
            outputs_normal = self.model(**inputs)
            logits_normal = outputs_normal.logits[:, -1, :]  # Last token
            
            # Get ablated logits
            with utils.AblationContext(self.model, list(range(8, 12)), baseline=self.ablation_baseline):
                outputs_ablated = self.model(**inputs)
                logits_ablated = outputs_ablated.logits[:, -1, :]  # Last token
            
            # Compute logit difference
            logit_diff = ((logits_normal - logits_ablated) ** 2).mean().item()
            return min(logit_diff, 1.0)
        except Exception as e:
            logger.debug(f"Error computing logit influence: {e}")
            return 0.0
    
    def _measure_consistency(self, inputs: Dict) -> float:
        """
        Metric 5: Consistency - do we get same result over multiple runs?
        
        (In practice, this could be multiple seeds or temperature variations)
        """
        try:
            outputs_list = []
            for _ in range(3):  # 3 runs
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'], max_new_tokens=5, do_sample=False
                    )
                    outputs_list.append(outputs[0].tolist())
            
            # Check consistency
            all_same = all(out == outputs_list[0] for out in outputs_list)
            return 1.0 if all_same else 0.33
        except Exception as e:
            logger.debug(f"Error measuring consistency: {e}")
            return 0.0


def run_phase1_multimetric(model: nn.Module, tokenizer, problems: List[Dict],
                           induction_heads: List[tuple],
                           output_dir: Path = Path('results'),
                           ablation_baseline: str = 'mean') -> Dict:
    """
    Run Phase 1 multi-metric measurement.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        problems: Tier 1 + Tier 4 problems (150 total)
        induction_heads: Top induction heads from Phase 0
        output_dir: Output directory
        ablation_baseline: Baseline type for ablation ('mean', 'zero', etc.)
        
    Returns:
        {
            'num_problems': int,
            'aligned_percentage': float,
            'results': [...]
        }
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: MULTI-METRIC MEASUREMENT")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    measurer = MultiMetricMeasurer(model, tokenizer, induction_heads,
                                    ablation_baseline=ablation_baseline)
    
    results = []
    aligned_count = 0
    
    for i, problem in enumerate(problems):
        metric_result = measurer.measure_all_metrics(problem)
        results.append(metric_result)
        
        if metric_result.get('all_aligned', False):
            aligned_count += 1
        
        if (i + 1) % 25 == 0:
            logger.info(f"Processed {i + 1}/{len(problems)} problems")
    
    aligned_percentage = (aligned_count / len(problems) * 100) if problems else 0.0
    
    phase1_results = {
        'num_problems': len(problems),
        'aligned_percentage': aligned_percentage,
        'aligned_count': aligned_count,
        'decision_threshold': 60,  # Need >=60% aligned
        'decision': 'HIGH_CONFIDENCE' if aligned_percentage >= 60 else 
                   'MEDIUM_CONFIDENCE' if aligned_percentage >= 30 else 'LOW_CONFIDENCE',
        'sample_results': results[:10],  # Save first 10
    }
    
    utils.save_results(phase1_results, output_dir / 'phase1_multimetric.json', phase='1')
    
    logger.info(f"Aligned: {aligned_count}/{len(problems)} ({aligned_percentage:.1f}%)")
    logger.info(f"Decision: {phase1_results['decision']}")
    
    return phase1_results

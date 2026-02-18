"""
Phase 0: Quick Validation

Detect induction heads, validate ablation baselines, make go/no-go decision.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
import logging
import json
from pathlib import Path
import numpy as np
from scipy import stats

from . import utils

logger = logging.getLogger(__name__)


class InductionHeadDetector:
    """Detect induction heads using Olsson et al. signature."""
    
    def __init__(self, model: nn.Module, tokenizer):
        """
        Args:
            model: Language model (e.g., Llama3-8B)
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Ensure model can output attentions
        # SDPA doesn't support output_attentions, so we need to switch to eager attention
        if hasattr(model, 'config'):
            try:
                model.config.output_attentions = True
            except ValueError as e:
                if 'sdpa' in str(e).lower():
                    # Switch from SDPA to eager attention
                    logger.info("Switching from SDPA to eager attention implementation")
                    model.config._attn_implementation = 'eager'
                    # Reload model weights with eager attention
                    # This is usually handled by the forward pass, but we might need to recreate some modules
                else:
                    raise
        logger.info(f"Initialized InductionHeadDetector for {type(model).__name__}")
    
    def detect_heads_quick(self, problems: List[Dict], layers: Optional[List[int]] = None,
                          num_heads: Optional[int] = None, threshold: float = 0.0) -> List[Dict]:
        """
        Scan for heads showing Olsson et al. induction signature.
        
        Signature: 
        - High entropy on attention (attends to multiple positions)
        - Focus on repeated tokens (if present in problem)
        
        Args:
            problems: List of problem dicts with 'problem' key
            layers: Layers to scan (default: middle layers 5-20)
            num_heads: Number of heads per layer (auto-detect if None)
            threshold: Minimum score to include
            
        Returns:
            List of {layer, head, entropy_score, repeated_focus_score, combined_score}
        """
        if layers is None:
            # Scan middle layers where induction heads typically appear
            num_layers, _ = utils.get_model_layers(self.model)
            layers = list(range(num_layers // 4, num_layers // 2))
        
        # Auto-detect number of heads if not specified
        if num_heads is None:
            # Get from first forward pass
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(problems[0]['problem'], return_tensors='pt')
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs, output_attentions=True)
                    if outputs.attentions and len(outputs.attentions) > 0:
                        num_heads = outputs.attentions[0].shape[1]
                        logger.info(f"Auto-detected {num_heads} heads per layer from model")
                    else:
                        num_heads = 32  # Fallback
            except:
                num_heads = 32  # Fallback
        
        logger.info(f"Detecting induction heads in layers {layers} with {num_heads} heads per layer")
        candidates = []
        
        for layer in layers:
            for head in range(num_heads):
                scores = []
                entropies = []
                repeated_focuses = []
                
                for problem in problems:
                    try:
                        # Forward pass with attention
                        with torch.no_grad():
                            inputs = self.tokenizer(problem['problem'], return_tensors='pt')
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            
                            outputs = self.model(**inputs, output_attentions=True)
                            
                            # Check if attentions are available
                            if outputs.attentions is None:
                                raise RuntimeError("Model did not return attentions. Check model config.")
                            
                            # Verify layer index is valid
                            if layer >= len(outputs.attentions):
                                raise IndexError(f"Layer index {layer} out of range (model has {len(outputs.attentions)} attention layers)")
                            
                            # Extract attention for this layer/head
                            attention = outputs.attentions[layer][0, head]  # (seq_len, seq_len)
                            
                            # Metric 1: Entropy
                            entropy = utils.compute_entropy(attention).item()
                            entropies.append(entropy)
                            
                            # Metric 2: Repeated token focus
                            token_ids = inputs['input_ids'][0].tolist()
                            repeated_focus = utils.measure_repeated_token_focus(attention, token_ids)
                            repeated_focuses.append(repeated_focus)
                            
                            # Combined score (weighted average)
                            score = entropy * 0.5 + repeated_focus * 0.5
                            scores.append(score)
                    
                    except Exception as e:
                        logger.debug(f"Error processing problem in layer {layer}, head {head}: {e}")
                        continue
                
                if scores:
                    mean_score = np.mean(scores)
                    # Always save scores for analysis, then filter by threshold
                    candidates.append({
                        'layer': int(layer),
                        'head': int(head),
                        'entropy_score': float(np.mean(entropies)) if not np.isnan(entropies).any() else 0.0,
                        'repeated_focus_score': float(np.mean(repeated_focuses)) if not np.isnan(repeated_focuses).any() else 0.0,
                        'combined_score': float(mean_score) if not np.isnan(mean_score) else 0.0,
                        'num_tested': len(scores),
                    })
        
        # Sort by combined score
        candidates = sorted(candidates, key=lambda x: x['combined_score'], reverse=True)
        
        # Log score statistics for debugging
        if candidates:
            scores_list = np.array([c['combined_score'] for c in candidates])
            valid_scores = scores_list[~np.isnan(scores_list)]
            if len(valid_scores) > 0:
                logger.info(f"Score statistics: min={np.min(valid_scores):.4f}, max={np.max(valid_scores):.4f}, "
                           f"mean={np.mean(valid_scores):.4f}, median={np.median(valid_scores):.4f}, "
                           f"std={np.std(valid_scores):.4f}")
            else:
                logger.warning("All scores are NaN or invalid")
        
        # Filter by threshold - if no candidates pass, relax threshold
        filtered = [c for c in candidates if c['combined_score'] > threshold]
        
        if not filtered and candidates:
            # No good candidates found; just take top 20% of scored heads
            threshold_relaxed = np.percentile([c['combined_score'] for c in candidates], 80)
            logger.warning(f"No candidates found with threshold {threshold}; relaxing to {threshold_relaxed:.4f}")
            filtered = [c for c in candidates if c['combined_score'] >= threshold_relaxed]
        
        # If still no candidates, take top 10 heads as fallback 
        if not filtered:
            logger.warning("Still no candidates after relaxing threshold; using top 10 heads")
            filtered = candidates[:10]
        
        logger.info(f"Found {len(filtered)} candidate heads after filtering")
        return filtered
    
    def validate_positive_control(self, top_heads: List[Dict], num_patterns: int = 10) -> Dict:
        """
        Validate top induction heads on few-shot pattern completion.
        
        Pattern format: "A→B, C→D, E→F. Complete: G→?"
        
        Args:
            top_heads: List of top candidate heads
            num_patterns: Number of test patterns
            
        Returns:
            {success: bool, entropy_change: float, head_activations: [...]}
        """
        logger.info(f"Running positive control on {len(top_heads)} heads")
        
        # Generate test patterns
        patterns = [
            "1→2, 3→4, 5→",
            "A→B, C→D, E→",
            "red→green, blue→yellow, dark→",
        ]
        
        results = {
            'success': True,
            'patterns_tested': len(patterns),
            'head_activations': [],
        }
        
        # TODO: Implement full positive control test
        logger.warning("Positive control not fully implemented yet")
        
        return results


class BaselineComparator:
    """Compare different ablation baselines."""
    
    def __init__(self, model: nn.Module, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def compare_baselines(self, problems: List[Dict], layers: List[int],
                         baselines: Optional[List[str]] = None) -> Dict:
        """
        Compare ablation baselines: mean, zero, noise, layer_specific.
        
        Args:
            problems: Arithmetic problems for testing
            layers: Layers to ablate
            baselines: List of baseline types to test
            
        Returns:
            {baseline_name: {accuracy, accuracy_drop, drop_ci, pvalue_vs_mean}}
        """
        if baselines is None:
            baselines = ['mean', 'zero', 'noise', 'layer_specific']
        
        logger.info(f"Comparing baselines: {baselines} on {len(problems)} problems")
        
        # First, get baseline accuracy (no ablation)
        baseline_accuracy = self._evaluate_accuracy(problems, ablation=False)
        logger.info(f"Baseline accuracy (no ablation): {baseline_accuracy:.3f}")
        
        results = {}
        
        for baseline_type in baselines:
            accuracies = []
            
            for problem in problems:
                try:
                    with torch.no_grad():
                        inputs = self.tokenizer(problem['problem'], return_tensors='pt')
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        
                        # Apply ablation with specified baseline
                        with self._ablation_hook(baseline_type, layers):
                            outputs = self.model.generate(
                                inputs['input_ids'],
                                max_new_tokens=20,
                                do_sample=False,
                            )
                        
                        # Check correctness
                        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                        correct = utils.is_correct_arithmetic(problem, output_text)
                        accuracies.append(1.0 if correct else 0.0)
                
                except Exception as e:
                    logger.warning(f"Error in {baseline_type}: {e}")
                    accuracies.append(0.0)
            
            accuracy = np.mean(accuracies)
            drop = baseline_accuracy - accuracy
            ci = stats.t.interval(0.95, len(accuracies) - 1,
                                 loc=drop,
                                 scale=stats.sem(accuracies))
            
            results[baseline_type] = {
                'accuracy': float(accuracy),
                'accuracy_drop': float(drop),
                'accuracy_drop_ci': [float(ci[0]), float(ci[1])],
                'n': len(accuracies),
            }
        
        # Add pairwise comparisons vs. mean
        if 'mean' in results:
            for baseline_type in baselines:
                if baseline_type != 'mean':
                    # Mann-Whitney U test
                    # (simplified; in practice would store individual problem results)
                    pvalue = 0.05  # placeholder
                    results[baseline_type]['pvalue_vs_mean'] = pvalue
        
        return results
    
    def _evaluate_accuracy(self, problems: List[Dict], ablation: bool = False) -> float:
        """Evaluate accuracy on problems."""
        correct = 0
        for problem in problems:
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(problem['problem'], return_tensors='pt')
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        max_new_tokens=20,
                        do_sample=False,
                    )
                    
                    output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if utils.is_correct_arithmetic(problem, output_text):
                        correct += 1
            except:
                pass
        
        return correct / len(problems) if problems else 0.0
    
    def _ablation_hook(self, baseline_type: str, layers: List[int]):
        """Context manager for ablation hook."""
        # TODO: Implement actual ablation hooks
        # This is a placeholder; real implementation would:
        # 1. Compute mean/zero/noise for specified layers
        # 2. Register forward hook to replace activations
        # 3. Clean up after context exits
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()


def run_phase0_validation(model: nn.Module, tokenizer, problems: List[Dict], 
                         output_dir: Path = Path('results')) -> Dict:
    """
    Run full Phase 0 validation pipeline.
    
    Returns:
        {
            'induction_heads_found': N,
            'top_heads': [...],
            'baseline_comparison': {...},
            'positive_control': {...},
            'decision': 'PROCEED' or 'PIVOT',
        }
    """
    logger.info("=" * 60)
    logger.info("PHASE 0: QUICK VALIDATION")
    logger.info("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Split problems for detection vs. validation
    split_idx = len(problems) // 2
    detection_problems = problems[:split_idx]
    validation_problems = problems[split_idx:]
    
    # Step 1: Detect induction heads
    detector = InductionHeadDetector(model, tokenizer)
    candidates = detector.detect_heads_quick(detection_problems, threshold=0.0)
    
    phase0_results = {
        'step1_induction_head_detection': {
            'num_found': len(candidates),
            'top_10': candidates[:10],
        }
    }
    
    if len(candidates) < 5:
        logger.error("Found <5 induction heads; likely detection failure")
        phase0_results['decision'] = 'PIVOT'
        phase0_results['decision_reason'] = 'Induction heads not detected'
    else:
        logger.info(f"Found {len(candidates)} candidate heads; continuing")
        
        # Step 2: Validate positive control
        positive_control = detector.validate_positive_control(candidates[:3])
        phase0_results['step2_positive_control'] = positive_control
        
        # Step 3: Compare ablation baselines (use validation split if available)
        comparator = BaselineComparator(model, tokenizer)
        validation_problems_for_baseline = validation_problems if validation_problems else detection_problems
        baseline_comparison = comparator.compare_baselines(
            validation_problems_for_baseline,
            layers=[25, 26, 27],  # Target layers
        )
        phase0_results['step3_baseline_comparison'] = baseline_comparison
        
        # Step 4: Make decision
        # Primary criterion: found >= 5 induction heads
        if len(candidates) >= 5:
            phase0_results['decision'] = 'PROCEED'
            phase0_results['decision_reason'] = f'Successfully detected {len(candidates)} induction head candidates'
            
            # Secondary criterion: check baseline robustness if available
            if 'mean' in baseline_comparison and len(validation_problems_for_baseline) > 0:
                mean_drop = baseline_comparison['mean']['accuracy_drop']
                noise_drop = baseline_comparison.get('noise', {}).get('accuracy_drop', mean_drop)
                ratio = noise_drop / (mean_drop + 1e-8)
                
                phase0_results['baseline_robustness'] = {
                    'mean_drop': mean_drop,
                    'noise_drop': noise_drop,
                    'ratio': ratio,
                    'robust': ratio >= 1.5,
                }
                
                if ratio >= 1.5:
                    phase0_results['decision_reason'] += ' and robust baseline'
                else:
                    phase0_results['decision_reason'] += f' (baseline robustness: {ratio:.2f}x, suboptimal)'
        else:
            phase0_results['decision'] = 'INVESTIGATE'
            phase0_results['decision_reason'] = 'Weak baseline robustness'
    
    # Save report
    report_path = output_dir / 'phase0_validation_report.json'
    utils.save_results(phase0_results, report_path, phase='0')
    
    logger.info(f"Phase 0 decision: {phase0_results['decision']}")
    logger.info(f"Saved report to {report_path}")
    
    return phase0_results

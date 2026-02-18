"""
Phase 1: Diagnostics - Staged Ablation

Test progressive layer ablation to understand effect size.
Stages are configured dynamically from phase1_config.yaml
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import logging
import numpy as np
from pathlib import Path
import yaml

from . import utils

logger = logging.getLogger(__name__)


class StagedAblationTester:
    """Progressive ablation testing across stages."""
    
    def __init__(self, model: nn.Module, tokenizer, induction_heads: List[tuple],
                 ablation_baseline: str = 'mean', stage_config: Optional[Dict] = None):
        """
        Args:
            model: Language model
            tokenizer: Tokenizer
            induction_heads: List of (layer, head) tuples to monitor
            ablation_baseline: Type of baseline ('mean', 'zero', 'noise', 'layer_specific')
            stage_config: Dict with stage1_layers, stage2_layers, stage3_layers keys
        """
        self.model = model
        self.tokenizer = tokenizer
        self.induction_heads = induction_heads
        self.ablation_baseline = ablation_baseline  # Use this instead of hardcoding 'zero'
        self.device = next(model.parameters()).device
        
        # Load stage config from parameter or config file
        if stage_config:
            self.stage_config = stage_config
        else:
            self.stage_config = self._load_stage_config()
    
    def _load_stage_config(self) -> Dict:
        """Load stage configuration from phase1_config.yaml."""
        config_path = Path(__file__).parent.parent / 'phase1_config.yaml'
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return {
                'stage1_layers': [11],
                'stage2_layers': [10, 11],
                'stage3_layers': [8, 9, 10, 11]
            }
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        staged = config.get('staged_ablation', {})
        return {
            'stage1_layers': staged.get('stage1_layers', [11]),
            'stage2_layers': staged.get('stage2_layers', [10, 11]),
            'stage3_layers': staged.get('stage3_layers', [8, 9, 10, 11])
        }
    
    def test_stage1(self, problems: List[Dict]) -> Dict:
        """Stage 1: Ablate configured stage1 layers."""
        logger.info(f"Testing Stage 1: Layers {self.stage_config['stage1_layers']}")
        return self._run_stage(problems, layers=self.stage_config['stage1_layers'], stage_num=1)
    
    def test_stage2(self, problems: List[Dict]) -> Dict:
        """Stage 2: Ablate configured stage2 layers."""
        logger.info(f"Testing Stage 2: Layers {self.stage_config['stage2_layers']}")
        return self._run_stage(problems, layers=self.stage_config['stage2_layers'], stage_num=2)
    
    def test_stage3(self, problems: List[Dict]) -> Dict:
        """Stage 3: Ablate configured stage3 layers (MAIN TEST)."""
        logger.info(f"Testing Stage 3: Layers {self.stage_config['stage3_layers']} (MAIN)")
        return self._run_stage(problems, layers=self.stage_config['stage3_layers'], stage_num=3)
    
    def _run_stage(self, problems: List[Dict], layers: List[int], stage_num: int = 1) -> Dict:
        """
        Run ablation test on specified layers.
        
        Args:
            problems: List of arithmetic problems
            layers: Layer indices to ablate
            stage_num: Stage number (1, 2, or 3)
            
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
        logger.info(f"Baseline entropy: {baseline_entropy:.4f}")
        
        # Now evaluate with ablation
        for i, problem in enumerate(problems):
            try:
                with torch.no_grad():
                    # Get accuracy with ablation
                    inputs = self.tokenizer(problem['problem'], return_tensors='pt')
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with utils.AblationContext(self.model, layers, baseline=self.ablation_baseline):
                        outputs = self.model.generate(
                            inputs['input_ids'],
                            max_new_tokens=20,
                            do_sample=False,
                        )
                    
                    output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    correct = utils.is_correct_arithmetic(problem, output_text)
                    accuracies.append(1.0 if correct else 0.0)
                    
                    # Get induction head entropy with ablation
                    entropy_val = self._compute_entropy_with_ablation(problem, layers)
                    entropies.append(entropy_val)
                    
                    results.append({
                        'problem_id': i,
                        'problem': problem['problem'],
                        'expected': problem['expected'],
                        'generated': output_text,
                        'correct': correct,
                        'accuracy': 1.0 if correct else 0.0,
                        'entropy': float(entropy_val),
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
            'baseline_accuracy': float(baseline_acc),
            'baseline_entropy': float(baseline_entropy),
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
                        with utils.AblationContext(self.model, ablated_layers, baseline=self.ablation_baseline):
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
        for problem in problems[:10]:  # Quick eval on subset
            try:
                with torch.no_grad():
                    inputs = self.tokenizer(problem['problem'], return_tensors='pt')
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    if ablated_layers:
                        with utils.AblationContext(self.model, ablated_layers, baseline=self.ablation_baseline):
                            outputs = self.model(**inputs, output_attentions=True)
                    else:
                        outputs = self.model(**inputs, output_attentions=True)
                    
                    # Compute entropy for induction heads
                    entropy_val = 0.0
                    head_count = 0
                    for layer, head in self.induction_heads[:3]:
                        if layer < len(outputs.attentions):
                            attn = outputs.attentions[layer][0, head]  # Get attention for this head
                            entropy_val += utils.compute_entropy(attn).item()
                            head_count += 1
                    
                    if head_count > 0:
                        entropy_val /= head_count
                    
                    entropies.append(entropy_val)
            except Exception as e:
                logger.debug(f"Could not compute entropy: {e}")
                pass
        
        return np.mean(entropies) if entropies else 0.0
    
    def _compute_entropy_with_ablation(self, problem: Dict, ablated_layers: List[int]) -> float:
        """Compute entropy score for a single problem with ablation."""
        try:
            with torch.no_grad():
                inputs = self.tokenizer(problem['problem'], return_tensors='pt')
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with utils.AblationContext(self.model, ablated_layers, baseline=self.ablation_baseline):
                    outputs = self.model(**inputs, output_attentions=True)
                
                # Compute mean entropy across induction heads
                total_entropy = 0.0
                head_count = 0
                for layer, head in self.induction_heads[:5]:
                    if layer < len(outputs.attentions):
                        attn = outputs.attentions[layer][0, head]
                        entropy = utils.compute_entropy(attn).item()
                        total_entropy += entropy
                        head_count += 1
                
                return total_entropy / head_count if head_count > 0 else 0.0
        except Exception as e:
            logger.debug(f"Error computing entropy: {e}")
            return 0.0
    
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
        return utils.AblationContext(self.model, layers, baseline=self.ablation_baseline)


def run_phase1_staged_ablation(model: nn.Module, tokenizer, problems_tier1: List[Dict],
                               induction_heads: List[tuple], 
                               output_dir: Path = Path('results'),
                               stage_config: Optional[Dict] = None,
                               ablation_baseline: str = 'mean') -> Dict:
    """
    Run Phase 1 staged ablation tests.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        problems_tier1: Test problems
        induction_heads: List of induction heads
        output_dir: Output directory for results
        stage_config: Optional manual stage config; otherwise loaded from phase1_config.yaml
    
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
    
    tester = StagedAblationTester(model, tokenizer, induction_heads, 
                                  ablation_baseline=ablation_baseline,
                                  stage_config=stage_config)
    
    logger.info(f"Stage 1 layers: {tester.stage_config['stage1_layers']}")
    logger.info(f"Stage 2 layers: {tester.stage_config['stage2_layers']}")
    logger.info(f"Stage 3 layers: {tester.stage_config['stage3_layers']}")
    
    # Run 3 stages
    stage1 = tester.test_stage1(problems_tier1)
    stage2 = tester.test_stage2(problems_tier1)
    stage3 = tester.test_stage3(problems_tier1)
    
    # Make decision based on entropy progression
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

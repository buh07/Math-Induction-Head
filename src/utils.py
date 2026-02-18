"""
Utility functions and classes for induction head experiment.

Includes:
- HookManager: Register/cleanup forward hooks
- ArithmeticDataset: Generate arithmetic problems
- Ablation helpers: Extract representations, apply ablations
- Metrics: Compute attention entropy, behavioral impact, etc.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_model_layers(model: nn.Module) -> Tuple[int, nn.ModuleList]:
    """
    Extract the number of layers and the layers themselves from a model.
    Handles different model architectures (Llama, GPT-2, etc.).
    
    Args:
        model: The language model
        
    Returns:
        Tuple of (num_layers: int, layers: nn.ModuleList)
        
    Raises:
        ValueError: If model architecture is not recognized
    """
    # Check for Llama architecture
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        return len(layers), layers
    
    # Check for GPT-2 architecture
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        return len(layers), layers
    
    # Check for generic transformer architecture
    if hasattr(model, 'layers'):
        layers = model.layers
        return len(layers), layers
    
    raise ValueError(
        f"Could not determine model architecture. "
        f"Model type: {type(model).__name__}. "
        f"Available attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:20]}"
    )


class HookManager:
    """Context manager for registering/cleaning up forward hooks."""
    
    def __init__(self):
        self.hooks = []
    
    def register_hook(self, module, hook_fn: Callable) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook on a module."""
        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        return handle
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all_hooks()


class ArithmeticDataset(Dataset):
    """Generate arithmetic problems for testing."""
    
    def __init__(self, num_problems: int = 100, min_val: int = 0, max_val: int = 1000, seed: int = 42):
        """
        Generate arithmetic problems.
        
        Args:
            num_problems: Number of problems to generate
            min_val: Minimum operand value
            max_val: Maximum operand value
            seed: Random seed for reproducibility
        """
        self.num_problems = num_problems
        self.min_val = min_val
        self.max_val = max_val
        self.seed = seed
        self.problems = self._generate_problems()
    
    def _generate_problems(self) -> List[Dict]:
        """Generate arithmetic problems."""
        np.random.seed(self.seed)
        problems = []
        for _ in range(self.num_problems):
            a = np.random.randint(self.min_val, self.max_val)
            b = np.random.randint(self.min_val, self.max_val)
            result = a + b
            problem = {
                'problem': f"{a} + {b} =",
                'answer': str(result),
                'a': a,
                'b': b,
                'expected': result,
            }
            problems.append(problem)
        return problems
    
    def __len__(self) -> int:
        return len(self.problems)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.problems[idx]
    
    def save(self, filepath: Path):
        """Save problems to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.problems, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> List[Dict]:
        """Load problems from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


def is_correct_arithmetic(problem: Dict, output: str) -> bool:
    """
    Check if model output correctly solves arithmetic problem.
    
    Args:
        problem: Problem dict with 'expected' key
        output: Model output string
        
    Returns:
        True if output contains correct answer
    """
    expected = str(problem['expected'])
    # Simple check: does output contain the expected number?
    return expected in output


def extract_attention_weights(model_output, layer: int, head: Optional[int] = None) -> torch.Tensor:
    """
    Extract attention weights from model output.
    
    Args:
        model_output: Model output with attentions
        layer: Which layer to extract
        head: Which head (None = all heads)
        
    Returns:
        Attention tensor of shape (seq_len, seq_len) or (num_heads, seq_len, seq_len)
    """
    if not hasattr(model_output, 'attentions') or model_output.attentions is None:
        raise ValueError("Model output does not contain attention weights")
    
    # attentions is tuple of (batch_size, num_heads, seq_len, seq_len) for each layer
    attention = model_output.attentions[layer][0]  # Remove batch dim
    
    if head is not None:
        attention = attention[head]
    
    return attention


def extract_mlp_activations(model: nn.Module, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract pre and post MLP activations.
    
    Args:
        model: Language model
        layer: Which layer
        
    Returns:
        (pre_activation, post_activation) tensors
    """
    # This is a stub; implementation depends on model architecture
    raise NotImplementedError("Model-specific implementation needed")


def compute_entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Shannon entropy of probability distribution.
    
    Args:
        probs: Probability tensor (attention weights)
        dim: Dimension to compute entropy over
        eps: Small value for numerical stability
        
    Returns:
        Entropy value (scalar or tensor)
    """
    probs = torch.clamp(probs, min=eps)
    entropy = -(probs * torch.log(probs)).sum(dim=dim)
    return entropy


def measure_repeated_token_focus(attention: torch.Tensor, problem_tokens: List[int]) -> float:
    """
    Measure attention focus on repeated tokens (induction head signature).
    
    Args:
        attention: Attention weights (seq_len, seq_len)
        problem_tokens: Token IDs in problem
        
    Returns:
        Score 0-1 indicating focus on repeated tokens
    """
    # Find repeated token positions
    seq_len = len(problem_tokens)
    repeated_positions = []
    
    for i in range(seq_len):
        for j in range(i):
            if problem_tokens[i] == problem_tokens[j]:
                repeated_positions.append((i, j))
    
    if not repeated_positions:
        return 0.0
    
    # Measure total attention to repeated token positions
    focus = 0.0
    for current_pos, previous_pos in repeated_positions:
        if current_pos < seq_len:
            focus += attention[current_pos, previous_pos].item()
    
    return min(focus / len(repeated_positions), 1.0)


def compute_behavioral_diff(logits1: torch.Tensor, logits2: torch.Tensor, method: str = "kl") -> float:
    """
    Compute behavioral difference between two logit distributions.
    
    Args:
        logits1: Original logits
        logits2: Ablated logits
        method: 'kl', 'js', 'l2', or 'jaccard'
        
    Returns:
        Scalar difference measure
    """
    probs1 = torch.softmax(logits1, dim=-1)
    probs2 = torch.softmax(logits2, dim=-1)
    
    if method == "kl":
        # KL divergence
        kl = (probs1 * (torch.log(probs1) - torch.log(probs2))).sum(dim=-1).mean()
        return kl.item()
    elif method == "js":
        # Jensen-Shannon divergence
        m = 0.5 * (probs1 + probs2)
        js = 0.5 * (probs1 * (torch.log(probs1) - torch.log(m))).sum(dim=-1)
        js += 0.5 * (probs2 * (torch.log(probs2) - torch.log(m))).sum(dim=-1)
        return js.mean().item()
    elif method == "l2":
        # L2 distance
        return ((probs1 - probs2) ** 2).sum(dim=-1).mean().sqrt().item()
    elif method == "jaccard":
        # Jaccard similarity (1 - Jaccard distance)
        intersection = (probs1 * probs2).sum(dim=-1)
        union = (probs1 + probs2 - probs1 * probs2).sum(dim=-1)
        jaccard = intersection / (union + 1e-8)
        return (1 - jaccard.mean()).item()
    else:
        raise ValueError(f"Unknown method: {method}")


class AblationConfig:
    """Configuration for ablation experiments."""
    
    def __init__(self, ablated_layers: List[int], baseline: str = "mean", 
                 induction_heads: Optional[List[Tuple[int, int]]] = None):
        """
        Args:
            ablated_layers: List of layer indices to ablate
            baseline: Type of ablation baseline ('mean', 'zero', 'noise', 'layer_specific')
            induction_heads: List of (layer, head) tuples to monitor
        """
        self.ablated_layers = ablated_layers
        self.baseline = baseline
        self.induction_heads = induction_heads or []
    
    def to_dict(self) -> Dict:
        return {
            'ablated_layers': self.ablated_layers,
            'baseline': self.baseline,
            'induction_heads': self.induction_heads,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'AblationConfig':
        return cls(**config_dict)


def save_results(results: Dict, filepath: Path, phase: str = "unknown"):
    """Save results to JSON file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        'phase': phase,
        'timestamp': str(np.datetime64('now')),
    }
    output = {'metadata': metadata, 'results': results}
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved results to {filepath}")


def load_results(filepath: Path) -> Dict:
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def setup_logging(log_level: int = logging.INFO):
    """Configure logging for experiment."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

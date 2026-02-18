"""
Example configuration and main entry point for induction head experiment.

Usage:
    python main.py --phase 0
    python main.py --phase 1 --config phase1_config.yaml
    python main.py --phase 2 --config phase2_config.yaml
    python main.py --phase 3
"""

import argparse
from pathlib import Path
import logging
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import src.utils as utils
import src.validation_suite as phase0
import src.staged_ablation as phase1_ablation
import src.multi_metric_measurement as phase1_metrics
import src.core_experiment as phase2
import src.statistical_validation as phase3

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


def setup_model(model_name: str = 'gpt2', use_quantization: bool = False):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    try:
        if use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map='auto' if torch.cuda.is_available() else None,
                torch_dtype=torch.float16,
                attn_implementation='eager',
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto' if torch.cuda.is_available() else None,
                attn_implementation='eager',
                trust_remote_code=True,
            )
            if not torch.cuda.is_available():
                model = model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        logger.info(f"✓ Model loaded: {model_name}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def run_phase0(model, tokenizer, output_dir: Path):
    """Run Phase 0: Quick validation."""
    logger.info("Starting Phase 0: Quick Validation")
    
    # Generate test problems - use sequence continuation (has repeated tokens for induction heads)
    # Mix arithmetic and sequence for robustness
    dataset = utils.ArithmeticDataset(num_problems=60, seed=42, problem_type='sequence')
    problems = dataset.problems
    
    # Run Phase 0
    results = phase0.run_phase0_validation(model, tokenizer, problems, output_dir)
    
    # Determine go/no-go
    if results['decision'] == 'PROCEED':
        logger.info("✓ Phase 0 PASSED: Proceed to Phase 1")
        return True, results
    else:
        logger.warning("✗ Phase 0 FAILED: Consider pivoting to alternative circuits")
        return False, results


def run_phase1(model, tokenizer, config: dict, output_dir: Path):
    """Run Phase 1: Diagnostics."""
    logger.info("Starting Phase 1: Diagnostics")
    
    # Load configuration from Phase 0
    config = load_config(Path('phase1_config.yaml'))
    induction_heads = config.get('induction_heads', [(7, 15), (8, 22)])
    staged_ablation_config = config.get('staged_ablation', {})
    ablation_baseline = config.get('ablation_baseline', 'mean')
    
    # Generate test problems
    dataset_t1 = utils.ArithmeticDataset(num_problems=100, seed=42)
    dataset_t4 = utils.ArithmeticDataset(num_problems=50, seed=43)
    
    # Run Phase 1a: Staged ablation (pass config explicitly)
    ablation_results = phase1_ablation.run_phase1_staged_ablation(
        model, tokenizer, dataset_t1.problems, induction_heads, output_dir,
        stage_config=staged_ablation_config,
        ablation_baseline=ablation_baseline
    )
    
    # Run Phase 1b: Multi-metric measurement
    all_problems = dataset_t1.problems + dataset_t4.problems
    metric_results = phase1_metrics.run_phase1_multimetric(
        model, tokenizer, all_problems, induction_heads, output_dir,
        ablation_baseline=ablation_baseline
    )
    
    logger.info("✓ Phase 1 COMPLETE: Ready for Phase 2 (or redesign if issues)")
    
    return {
        'staged_ablation': ablation_results,
        'multi_metrics': metric_results,
    }


def run_phase2(model, tokenizer, config: dict, output_dir: Path):
    """Run Phase 2: Core experiment."""
    logger.info("Starting Phase 2: Core Experiment")
    
    # Load configuration
    config = load_config(Path('phase2_config.yaml'))
    ablation_config = utils.AblationConfig(
        ablated_layers=config.get('ablated_layers', list(range(17, 32))),
        baseline=config.get('baseline', 'mean'),
        induction_heads=config.get('induction_heads', []),
    )
    
    # Generate test problems
    dataset_t1 = utils.ArithmeticDataset(num_problems=100, seed=42)
    dataset_t4 = utils.ArithmeticDataset(num_problems=50, seed=43)
    
    # Run Phase 2
    results = phase2.run_phase2_core_experiment(
        model, tokenizer, 
        dataset_t1.problems, dataset_t4.problems,
        ablation_config,
        output_dir
    )
    
    logger.info(f"✓ Phase 2 COMPLETE: Scenario {results['scenario']}")
    
    return results


def run_phase3(output_dir: Path):
    """Run Phase 3: Analysis & Publication."""
    logger.info("Starting Phase 3: Statistical Analysis")
    
    # Load results from Phase 2
    phase2_results_path = output_dir / 'phase2_core_experiment.json'
    if not phase2_results_path.exists():
        logger.error("Phase 2 results not found")
        return None
    
    phase2_results = utils.load_results(phase2_results_path)
    
    # Run Phase 3
    results = phase3.run_phase3_analysis(
        phase2_results['main_experiment'],
        phase2_results['controls'],
        output_dir
    )
    
    logger.info(f"✓ Phase 3 COMPLETE: {results['publication_tier']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Induction Head Experiment Runner')
    parser.add_argument('--phase', type=int, choices=[0, 1, 2, 3], required=True,
                       help='Which phase to run')
    parser.add_argument('--config', type=str, help='Config YAML file')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model name from HuggingFace')
    parser.add_argument('--quantize', action='store_true',
                       help='Use 8-bit quantization')
    parser.add_argument('--log-level', type=str, default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup
    utils.setup_logging(getattr(logging, args.log_level))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config(Path(args.config)) if args.config else {}
    
    # Load model only if needed (phases 0-2)
    if args.phase <= 2:
        model, tokenizer = setup_model(args.model, args.quantize)
    else:
        model, tokenizer = None, None
    
    # Run requested phase
    if args.phase == 0:
        success, results = run_phase0(model, tokenizer, output_dir)
        if not success:
            logger.warning("Phase 0 failed; stopping")
    
    elif args.phase == 1:
        run_phase1(model, tokenizer, config, output_dir)
    
    elif args.phase == 2:
        run_phase2(model, tokenizer, config, output_dir)
    
    elif args.phase == 3:
        run_phase3(output_dir)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()

"""Lightweight utilities for the rebooted induction-head project."""

from .config import ExperimentConfig, load_config_file
from .datasets import (
    ArithmeticDataset,
    DatasetBundle,
    DatasetSpec,
    TieredDatasetSuite,
    MultiOperationArithmeticDataset,
    GSMStyleDataset,
    generate_prompt_batch,
    load_tiered_suite,
)
from .logging_utils import RunLogger, create_run_manifest
from .hooks import AttentionHookConfig, HookManager, NeuronHookConfig
from .ablation import AblationStage, StagedAblationRunner
from .tokenization_diagnostics import TokenizationReport, analyze_prompts
from .hash_utils import hash_strings
from .evaluation import BaselineEvaluator, SweepResult, run_parameter_sweep
from .statistics import summarize
from .model_loader import load_local_model
from .experiment_runner import ExperimentRunner

__all__ = [
    "ExperimentConfig",
    "load_config_file",
    "ArithmeticDataset",
    "DatasetSpec",
    "DatasetBundle",
    "TieredDatasetSuite",
    "MultiOperationArithmeticDataset",
    "GSMStyleDataset",
    "load_tiered_suite",
    "generate_prompt_batch",
    "RunLogger",
    "create_run_manifest",
    "hash_strings",
    "summarize",
    "load_local_model",
    "ExperimentRunner",
    "AttentionHookConfig",
    "NeuronHookConfig",
    "HookManager",
    "AblationStage",
    "StagedAblationRunner",
    "TokenizationReport",
    "analyze_prompts",
    "BaselineEvaluator",
    "SweepResult",
    "run_parameter_sweep",
]

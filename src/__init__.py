"""Lightweight utilities for the rebooted induction-head project."""

from .config import ExperimentConfig, load_config_file
from .datasets import (
    ArithmeticDataset,
    DatasetBundle,
    DatasetSpec,
    TieredDatasetSuite,
    MultiOperationArithmeticDataset,
    GSMStyleDataset,
    GSM8KDataset,
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
from .operator_buckets import (
    OperatorBucketExample,
    OperatorBucketDataset,
    OperatorBucketSuite,
    generate_operator_bucket_dataset,
    generate_operator_bucket_suite,
    suite_diagnostics as operator_bucket_suite_diagnostics,
)
from .arithmetic_error_taxonomy import (
    assess_prediction as assess_arithmetic_prediction,
    summarize_bucket_predictions as summarize_arithmetic_bucket_predictions,
)
from .model_introspection import (
    locate_layers,
    get_attention_module,
    get_mlp_module,
    infer_head_count,
)
from .parser_audit import (
    ParserAuditSample,
    collect_parser_audit_samples_from_intervention_runs,
    build_parser_audit_report,
)
from .power_analysis import (
    required_n_per_arm_two_proportion,
    build_power_analysis_report,
)
from .runtime_batch_autotune import (
    BatchAutotuneConfig,
    autotune_batch_size,
)

__all__ = [
    "ExperimentConfig",
    "load_config_file",
    "ArithmeticDataset",
    "DatasetSpec",
    "DatasetBundle",
    "TieredDatasetSuite",
    "MultiOperationArithmeticDataset",
    "GSMStyleDataset",
    "GSM8KDataset",
    "load_tiered_suite",
    "generate_prompt_batch",
    "RunLogger",
    "create_run_manifest",
    "hash_strings",
    "summarize",
    "load_local_model",
    "ExperimentRunner",
    "OperatorBucketExample",
    "OperatorBucketDataset",
    "OperatorBucketSuite",
    "generate_operator_bucket_dataset",
    "generate_operator_bucket_suite",
    "operator_bucket_suite_diagnostics",
    "assess_arithmetic_prediction",
    "summarize_arithmetic_bucket_predictions",
    "locate_layers",
    "get_attention_module",
    "get_mlp_module",
    "infer_head_count",
    "ParserAuditSample",
    "collect_parser_audit_samples_from_intervention_runs",
    "build_parser_audit_report",
    "required_n_per_arm_two_proportion",
    "build_power_analysis_report",
    "BatchAutotuneConfig",
    "autotune_batch_size",
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

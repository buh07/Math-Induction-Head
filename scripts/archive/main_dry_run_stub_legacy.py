"""
Minimal entry point for the rebuilt induction-head experiments.

Infrastructure goals:
- exercise configuration loading
- generate reproducible prompt batches
- list tiered datasets + hashes
- capture baseline stability reports before sweeps
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

from src import (
    ExperimentConfig,
    BaselineEvaluator,
    RunLogger,
    TieredDatasetSuite,
    load_tiered_suite,
    create_run_manifest,
    generate_prompt_batch,
    load_config_file,
    run_parameter_sweep,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infrastructure stub for upcoming induction-head experiments."
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for any artifacts produced by future implementations.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config overriding defaults.",
    )
    parser.add_argument(
        "--list-tiers",
        action="store_true",
        help="List dataset tiers and hashes, then continue.",
    )
    parser.add_argument(
        "--baseline-tier",
        default=None,
        help="Run baseline stability evaluation on the specified tier.",
    )
    parser.add_argument(
        "--baseline-runs",
        type=int,
        default=3,
        help="Number of baseline repetitions to estimate stability.",
    )
    parser.add_argument(
        "--run-sweep",
        action="store_true",
        help="Execute a placeholder parameter sweep (for automation testing).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_data = load_config_file(args.config) if args.config else {}
    exp_config = ExperimentConfig.from_dict(config_data)

    log_level = getattr(logging, exp_config.log_level.upper(), logging.INFO)
    logger = RunLogger(level=log_level).configure()

    os.environ["CUDA_VISIBLE_DEVICES"] = exp_config.devices
    logger.info("Pinned CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])
    logger.info("Running infrastructure dry-run with config: %s", exp_config)

    prompts = generate_prompt_batch(
        num_prompts=exp_config.problem_count,
        operand_range=(0, 100),
        seed=exp_config.seed,
    )

    suite: TieredDatasetSuite = load_tiered_suite(exp_config.seed)
    if args.list_tiers:
        for name, dataset_hash in suite.hashes().items():
            logger.info("Tier %s -> hash %s", name, dataset_hash)

    manifest_metadata = {
        "devices": exp_config.devices,
        "dataset_hashes": suite.hashes(),
    }

    if args.baseline_tier:
        bundle = suite.get(args.baseline_tier)

        def _evaluation_fn(prompts_list: List[str], run_seed: int) -> float:
            # Placeholder metric: normalized prompt length variance with deterministic seed.
            total_chars = sum(len(p) for p in prompts_list) + run_seed
            return (total_chars % 1000) / 1000.0

        evaluator = BaselineEvaluator(_evaluation_fn)
        baseline_report = evaluator.run(
            bundle, repeats=max(1, args.baseline_runs), seed_offset=exp_config.seed
        )
        logger.info(
            "Baseline %s -> mean %.4f ± %.4f",
            baseline_report.dataset_name,
            baseline_report.mean_score,
            baseline_report.std_dev,
        )
        manifest_metadata["baseline_report"] = {
            "dataset": baseline_report.dataset_name,
            "hash": baseline_report.dataset_hash,
            "scores": baseline_report.scores,
            "mean": baseline_report.mean_score,
            "std": baseline_report.std_dev,
        }

    if args.run_sweep:
        grid = {
            "attention_scale": [0.8, 0.9, 1.0],
            "neuron_scale": [0.8, 1.0],
        }

        def _sweep_fn(params: dict[str, float]) -> float:
            return 0.5 * params["attention_scale"] + 0.5 * params["neuron_scale"]

        sweep_results = run_parameter_sweep(grid, _sweep_fn)
        logger.info("Parameter sweep evaluated %d configurations", len(sweep_results))
        manifest_metadata["sweep_results"] = [
            {"params": result.params, "score": result.score}
            for result in sweep_results
        ]

    manifest_path = create_run_manifest(
        output_path, exp_config.to_dict(), extras=manifest_metadata
    )
    logger.info("Wrote run manifest to %s", manifest_path)
    logger.info("Generated %d prompts (dry-run only)", len(prompts))


if __name__ == "__main__":
    main()

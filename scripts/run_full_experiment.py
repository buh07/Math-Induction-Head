#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from src import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-part induction-head experiments.")
    parser.add_argument(
        "--plan",
        default="configs/experiment_plan.yaml",
        help="Path to experiment plan YAML.",
    )
    parser.add_argument(
        "--model-cache",
        default="../LLM Second-Order Effects/models",
        help="Cache directory containing local HF models.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/experiments",
        help="Directory to store experiment outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runner = ExperimentRunner(
        plan_path=Path(args.plan),
        model_cache_dir=Path(args.model_cache),
        results_dir=Path(args.results_dir),
    )
    runner.run()


if __name__ == "__main__":
    main()

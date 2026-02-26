#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-part induction-head experiments.")
    parser.add_argument(
        "--plan",
        default="configs/phase1/experiment_plan.yaml",
        help="Path to experiment plan YAML.",
    )
    parser.add_argument(
        "--model-cache",
        default="../LLM Second-Order Effects/models",
        help="Cache directory containing local HF models.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/phase1/failed_or_partial/experiments",
        help="Directory to store experiment outputs.",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Override CUDA_VISIBLE_DEVICES for all models in the plan (e.g. '5,6,7').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runner = ExperimentRunner(
        plan_path=Path(args.plan),
        model_cache_dir=Path(args.model_cache),
        results_dir=Path(args.results_dir),
        override_devices=args.devices,
    )
    runner.run()


if __name__ == "__main__":
    main()

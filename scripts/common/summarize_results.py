#!/usr/bin/env python3
"""
Utility script to compute statistical summaries from stored JSON results.

Expected input format:
{
  "baseline": [0.8, 0.81, 0.79],
  "intervention": [0.84, 0.86, 0.85]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.statistics import summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment results.")
    parser.add_argument("input", help="Path to JSON file with baseline/intervention scores.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write summary JSON (defaults to stdout).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    parser.add_argument(
        "--samples", type=int, default=1000, help="Number of bootstrap samples."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input).read_text())
    summary = summarize(
        baseline_scores=data["baseline"],
        intervention_scores=data["intervention"],
        num_bootstrap=args.samples,
        seed=args.seed,
    )
    output = summary.to_dict()
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

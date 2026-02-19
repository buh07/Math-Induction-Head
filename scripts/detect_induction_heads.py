#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.induction_detection import detect_induction_heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect induction heads for a model.")
    parser.add_argument("--model", required=True, help="Model name or path.")
    parser.add_argument("--cache-dir", default="../LLM Second-Order Effects/models/hub")
    parser.add_argument("--output", default="results/induction_heads.json")
    parser.add_argument("--prompts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    result = detect_induction_heads(
        model_name=args.model,
        cache_dir=args.cache_dir,
        prompt_count=args.prompts,
        seed=args.seed,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

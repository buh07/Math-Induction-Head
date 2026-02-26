#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.induction_detection import (
    aggregate_detection_runs,
    detect_induction_heads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect induction heads for a model.")
    parser.add_argument("--model", required=True, help="Model name or path.")
    parser.add_argument("--cache-dir", default="../LLM Second-Order Effects/models/hub")
    parser.add_argument("--output", default="results/phase1/failed_or_partial/induction_heads.json")
    parser.add_argument(
        "--prompts",
        type=int,
        default=50,
        help="Number of prompts to generate when using a synthetic suite or no prompt file.",
    )
    parser.add_argument(
        "--prompt-suite",
        choices=[
            "synthetic_repeat",
            "synthetic_repeat_numeric",
            "synthetic_negative",
            "gsm8k_plain",
            "gsm8k_cot",
            "custom",
        ],
        default=None,
        help="Named prompt suite. Use custom with --prompt-file to preserve suite metadata.",
    )
    parser.add_argument("--prompt-file", type=Path, help="Optional text file containing prompts (one per line).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-list", type=str, help="Comma-separated seeds; outputs an aggregate JSON with repeated runs.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--strict-head-hooks",
        dest="strict_head_hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require targeted head hooks to error if head decomposition fails.",
    )
    parser.add_argument(
        "--effect-token-policy",
        choices=["baseline_argmax", "explicit_copy_target"],
        default="baseline_argmax",
    )
    parser.add_argument("--metrics", choices=["basic", "causal", "full"], default="full")
    parser.add_argument("--save-per-prompt-effects", action="store_true")
    parser.add_argument("--epsilon", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    custom_prompts = None
    if args.prompt_file:
        text = args.prompt_file.read_text(encoding="utf-8")
        custom_prompts = [line.strip() for line in text.splitlines() if line.strip()]
    prompt_suite = args.prompt_suite
    if prompt_suite is None and args.prompt_file is not None:
        prompt_suite = "custom"

    seeds: List[int]
    if args.seed_list:
        seeds = [int(part.strip()) for part in args.seed_list.split(",") if part.strip()]
        if not seeds:
            raise ValueError("No valid seeds parsed from --seed-list")
    else:
        seeds = [args.seed]

    runs = []
    for seed in seeds:
        result = detect_induction_heads(
            model_name=args.model,
            cache_dir=args.cache_dir,
            prompt_count=args.prompts,
            seed=seed,
            prompts=custom_prompts,
            prompt_suite=prompt_suite,
            batch_size=args.batch_size,
            strict_head_hooks=args.strict_head_hooks,
            effect_token_policy=args.effect_token_policy,
            metrics_mode=args.metrics,
            epsilon=args.epsilon,
            save_per_prompt_effects=args.save_per_prompt_effects,
        )
        runs.append(result)

    if len(runs) == 1:
        payload = runs[0]
    else:
        payload = aggregate_detection_runs(runs)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

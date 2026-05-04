#!/usr/bin/env python3
"""Audit parser behavior from Phase 2 intervention artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.parser_audit import (  # noqa: E402
    build_parser_audit_report,
    collect_parser_audit_samples_from_intervention_runs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit parser behavior on Phase 2 intervention outputs.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to a Phase 2 run directory containing phase2_interventions.json.",
    )
    parser.add_argument(
        "--interventions-file",
        default="phase2_interventions.json",
        help="Intervention summary filename inside run dir.",
    )
    parser.add_argument(
        "--output",
        default="parser_audit.json",
        help="Output JSON filename (relative to run dir if not absolute).",
    )
    parser.add_argument(
        "--per-dataset-limit",
        type=int,
        default=16,
        help="Max sampled predictions per dataset/condition for audit.",
    )
    parser.add_argument(
        "--adjudication-cap",
        type=int,
        default=64,
        help="Max disagreement samples stored for manual review.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    interventions_path = run_dir / args.interventions_file
    if not interventions_path.exists():
        raise FileNotFoundError(f"Missing interventions artifact: {interventions_path}")

    payload = _load_json(interventions_path)
    runs = payload.get("runs")
    if not isinstance(runs, dict):
        raise ValueError(f"Expected 'runs' mapping in {interventions_path}")

    samples = collect_parser_audit_samples_from_intervention_runs(
        runs,
        per_dataset_limit=int(args.per_dataset_limit),
    )
    report = build_parser_audit_report(
        samples,
        source_label=str(interventions_path.relative_to(run_dir)),
        adjudication_cap=int(args.adjudication_cap),
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = run_dir / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()

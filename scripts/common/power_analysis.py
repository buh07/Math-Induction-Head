#!/usr/bin/env python3
"""Generate power-analysis report from preregistration config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.power_analysis import build_power_analysis_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2 power-analysis report.")
    parser.add_argument("--prereg", required=True, help="Path to preregistration YAML/JSON.")
    parser.add_argument(
        "--dataset-manifest",
        default=None,
        help="Optional dataset manifest JSON for observed counts.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path.",
    )
    return parser.parse_args()


def _load_mapping(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        parsed = yaml.safe_load(text)
    else:
        parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected mapping in {path}")
    return parsed


def main() -> None:
    args = parse_args()
    prereg_path = Path(args.prereg).resolve()
    prereg = _load_mapping(prereg_path)
    dataset_manifest = None
    if args.dataset_manifest:
        dataset_manifest = _load_mapping(Path(args.dataset_manifest).resolve())
    report = build_power_analysis_report(prereg, dataset_manifest=dataset_manifest)
    report["preregistration_path"] = str(prereg_path)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(str(output_path))


if __name__ == "__main__":
    main()

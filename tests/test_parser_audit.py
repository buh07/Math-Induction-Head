import json
from pathlib import Path

from src.parser_audit import (
    ParserAuditSample,
    build_parser_audit_report,
    collect_parser_audit_samples_from_intervention_runs,
)


def test_build_parser_audit_report_detects_disagreement_and_accuracy_delta():
    samples = [
        ParserAuditSample(
            source_run="addition::attention_heads:K5:top",
            dataset="addition__no_carry",
            operator="addition",
            bucket="no_carry",
            prompt="Q",
            output="The answer is 42.",
            target=42,
        ),
        ParserAuditSample(
            source_run="addition::attention_heads:K5:top",
            dataset="addition__single_carry",
            operator="addition",
            bucket="single_carry",
            prompt="Q",
            output="x = 11\n#### 13",
            target=13,
        ),
        ParserAuditSample(
            source_run="addition::attention_heads:K5:top",
            dataset="addition__single_carry",
            operator="addition",
            bucket="single_carry",
            prompt="Q",
            output="No numeric output",
            target=7,
        ),
    ]

    report = build_parser_audit_report(samples, source_label="phase2_interventions.json", adjudication_cap=8)
    assert report["schema_version"] == "parser_audit_v1"
    assert report["sample_count"] == 3
    assert report["target_available_count"] == 3
    assert 0.0 <= report["ambiguity_rate"] <= 1.0
    assert "strict_minus_default_accuracy" in report["parse_metrics"]["delta"]


def test_collect_parser_audit_samples_from_intervention_runs_reads_prediction_samples():
    payload = {
        "addition::attention_heads": {
            "results": [
                {
                    "condition": {"component_set_name": "K5:top"},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "bucket": "no_carry",
                            "prediction_samples": [
                                {
                                    "prompt": "1+1",
                                    "output": "2",
                                    "target": 2,
                                }
                            ],
                        }
                    },
                }
            ]
        }
    }
    samples = collect_parser_audit_samples_from_intervention_runs(payload, per_dataset_limit=4)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.dataset == "addition__no_carry"
    assert sample.output == "2"
    assert sample.target == 2


def test_collect_parser_audit_samples_reads_generated_fallback_key():
    payload = {
        "addition::attention_heads": {
            "results": [
                {
                    "condition": {"component_set_name": "K5:top"},
                    "datasets": {
                        "addition__no_carry": {
                            "operator": "addition",
                            "bucket": "no_carry",
                            "prediction_samples": [
                                {
                                    "prompt": "1+1",
                                    "generated": "2",
                                    "target": 2,
                                }
                            ],
                        }
                    },
                }
            ]
        }
    }
    samples = collect_parser_audit_samples_from_intervention_runs(payload, per_dataset_limit=4)
    assert len(samples) == 1
    assert samples[0].output == "2"


def test_audit_parser_behavior_cli_writes_output(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    interventions = {
        "schema_version": "phase2_interventions_summary_v1",
        "runs": {
            "addition::attention_heads": {
                "results": [
                    {
                        "condition": {"component_set_name": "K5:top"},
                        "datasets": {
                            "addition__no_carry": {
                                "operator": "addition",
                                "bucket": "no_carry",
                                "prediction_samples": [
                                    {
                                        "prompt": "1+1",
                                        "output": "2",
                                        "target": 2,
                                    }
                                ],
                            }
                        },
                    }
                ]
            }
        },
    }
    (run_dir / "phase2_interventions.json").write_text(json.dumps(interventions), encoding="utf-8")

    from scripts.common.audit_parser_behavior import main as cli_main
    import sys

    argv = sys.argv
    try:
        sys.argv = [
            "audit_parser_behavior.py",
            "--run-dir",
            str(run_dir),
            "--output",
            "parser_audit.json",
        ]
        cli_main()
    finally:
        sys.argv = argv

    out = json.loads((run_dir / "parser_audit.json").read_text(encoding="utf-8"))
    assert out["schema_version"] == "parser_audit_v1"
    assert out["sample_count"] == 1

import json

from main import main, parse_args


def test_parse_args_allows_overrides(tmp_path):
    args = parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--config",
            "missing.yaml",
            "--list-tiers",
            "--baseline-tier",
            "tier1_in_distribution",
            "--baseline-runs",
            "2",
            "--run-sweep",
        ]
    )
    assert args.output_dir == str(tmp_path)
    assert args.config == "missing.yaml"
    assert args.list_tiers is True
    assert args.baseline_tier == "tier1_in_distribution"
    assert args.baseline_runs == 2
    assert args.run_sweep is True


def test_main_creates_manifest(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "artifacts"
    main(
        [
            "--output-dir",
            str(output_dir),
            "--baseline-tier",
            "tier1_in_distribution",
            "--baseline-runs",
            "2",
            "--run-sweep",
        ]
    )
    manifest = output_dir / "run_manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text())
    assert "metadata" in data
    assert "baseline_report" in data["metadata"]
    assert "sweep_results" in data["metadata"]

import json

from src.power_analysis import build_power_analysis_report, required_n_per_arm_two_proportion


def test_required_n_per_arm_two_proportion_increases_for_smaller_effect():
    n_large = required_n_per_arm_two_proportion(
        baseline_rate=0.4,
        effect_size=0.1,
        alpha=0.05,
        power=0.8,
    )
    n_small = required_n_per_arm_two_proportion(
        baseline_rate=0.4,
        effect_size=0.05,
        alpha=0.05,
        power=0.8,
    )
    assert n_small > n_large > 0


def test_build_power_analysis_report_uses_primary_grid_and_manifest_counts():
    prereg = {
        "schema_version": "phase2_preregistration_v1",
        "alpha": 0.05,
        "target_power": 0.8,
        "minimum_effect_size_of_interest": {"delta_vs_random_accuracy_all": 0.05},
        "primary_comparison_grid": {
            "interventions": ["ablation", "amplification"],
            "k_values": [5, 10],
            "scales": [0.0, 1.25],
        },
        "multiplicity_policy": {"method": "bh_fdr"},
        "planned_sample_sizes": {
            "counts_per_bucket": 128,
            "operators": {"addition": 128, "subtraction": 128},
        },
        "assumptions": {"baseline_accuracy": 0.4},
    }
    manifest = {
        "counts_by_operator": {
            "addition": 384,
            "subtraction": 384,
        }
    }
    report = build_power_analysis_report(prereg, dataset_manifest=manifest)
    assert report["schema_version"] == "power_analysis_report_v1"
    assert report["n_primary_comparisons"] == 8
    cmp_row = report["comparisons"][0]
    assert cmp_row["metric"] == "delta_vs_random_accuracy_all"
    operators = {row["operator"] for row in cmp_row["operator_coverage"]}
    assert operators == {"addition", "subtraction"}


def test_power_analysis_cli_writes_report(tmp_path):
    prereg = {
        "schema_version": "phase2_preregistration_v1",
        "minimum_effect_size_of_interest": {"delta_vs_random_accuracy_all": 0.05},
        "primary_comparison_grid": {
            "interventions": ["ablation"],
            "k_values": [5],
            "scales": [0.0],
        },
        "planned_sample_sizes": {"counts_per_bucket": 64},
    }
    prereg_path = tmp_path / "prereg.json"
    prereg_path.write_text(json.dumps(prereg), encoding="utf-8")
    out_path = tmp_path / "power.json"

    from scripts.common.power_analysis import main as cli_main
    import sys

    argv = sys.argv
    try:
        sys.argv = [
            "power_analysis.py",
            "--prereg",
            str(prereg_path),
            "--output",
            str(out_path),
        ]
        cli_main()
    finally:
        sys.argv = argv

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "power_analysis_report_v1"

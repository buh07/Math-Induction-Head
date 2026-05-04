import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "common" / "audit_phase2_legacy_artifacts.py"


def test_legacy_audit_marks_readiness_false_when_required_gates_fail(tmp_path):
    run_dir = tmp_path / "results" / "phase2" / "run_v1"
    run_dir.mkdir(parents=True)
    gate = {
        "schema_version": "phase2_operator_bottleneck_gate_summary_v1",
        "phases": {
            "dataset_bucket_gate": {"passes": True},
            "localization_validity_gate": {
                "passes": True,
                "checks": [{"effect_nonzero_rate_max": 0.0, "answer_token_prob_delta_abs_max": 1e-9}],
            },
            "operator_specificity_gate": {
                "passes": True,
                "checks": [
                    {
                        "run": "addition::mlp_neurons",
                        "target_operator": "addition",
                        "passes": False,
                        "best_target": {"condition": "K5:bottom"},
                    }
                ],
            },
            "intervention_sanity_gate": {"passes": False},
            "cot_gating_evidence_gate": {"passes": False},
        },
        "overall": {"ready_for_multimodel_next_tranche": True},
    }
    (run_dir / "phase2_gate_summary.json").write_text(json.dumps(gate), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--results-root",
            str(tmp_path / "results" / "phase2"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr

    sidecar = json.loads((run_dir / "legacy_audit.json").read_text(encoding="utf-8"))
    assert sidecar["audited_ready_for_multimodel"] is False
    assert "required_gate_failed:cot_gating_evidence_gate" in sidecar["blocking_reasons"]
    assert "legacy_readiness_true_despite_failed_required_gates" in sidecar["known_legacy_defects"]

    index = json.loads((tmp_path / "results" / "phase2" / "legacy_audit_index.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == "legacy_audit_index_v1"
    assert len(index["runs"]) == 1


def test_legacy_audit_fail_if_v1_missing_sidecar_flag(tmp_path):
    run_dir = tmp_path / "results" / "phase2" / "run_v1"
    run_dir.mkdir(parents=True)
    gate = {
        "schema_version": "phase2_operator_bottleneck_gate_summary_v1",
        "phases": {},
        "overall": {},
    }
    (run_dir / "phase2_gate_summary.json").write_text(json.dumps(gate), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--results-root",
            str(tmp_path / "results" / "phase2"),
            "--fail-if-v1-missing-sidecar",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "Missing legacy_audit.json" in proc.stderr or "Missing legacy_audit.json" in proc.stdout

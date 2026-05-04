import json
import subprocess
import sys
from pathlib import Path

from scripts.phase2.run_operator_bottleneck_suite import _split_target_operator_datasets_for_selection_eval
from src.operator_buckets import generate_operator_bucket_suite


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "phase2" / "run_operator_bottleneck_suite.py"
CONFIG = ROOT / "configs" / "phase2" / "operator_buckets_llama3.yaml"
FULL_CONFIG = ROOT / "configs" / "phase2" / "operator_buckets_llama3_full_operators.yaml"


def _run(
    tmp_path: Path,
    stage: str,
    *,
    scaffold_gpu_stages: bool = False,
    config: Path = CONFIG,
    extra_args: list[str] | None = None,
):
    out = tmp_path / f"phase2_{stage}"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--stage",
        stage,
        "--smoke",
        "--dataset-config",
        str(config),
        "--output-root",
        str(out),
        "--low-cpu-mode",
        "--max-cpu-threads",
        "1",
    ]
    if scaffold_gpu_stages:
        cmd.append("--scaffold-gpu-stages")
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return proc, out


def test_phase2_orchestrator_datasets_stage_writes_manifest_and_gate_summary(tmp_path):
    proc, out = _run(tmp_path, "datasets")
    assert proc.returncode == 0, proc.stderr
    gate = json.loads((out / "phase2_gate_summary.json").read_text())
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    diagnostics = json.loads((out / "dataset_diagnostics.json").read_text())
    assert gate["schema_version"] == "phase2_operator_bottleneck_gate_summary_v2"
    assert gate["phases"]["dataset_bucket_gate"]["passes"] is True
    assert manifest["schema_version"] == "operator_bucket_suite_v1"
    assert diagnostics["schema_version"] == "dataset_diagnostics_v1"
    assert (out / "preregistration_used.json").exists()
    assert (out / "power_analysis_report.json").exists()
    assert (out / "replication_protocol.md").exists()
    assert (out / "run_manifest.json").exists()


def test_phase2_orchestrator_smoke_adjusts_min_valid_target_count(tmp_path):
    proc, out = _run(tmp_path, "datasets")
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out / "run_manifest.json").read_text())
    effective = manifest["effective_config"]
    assert effective["localization"]["min_valid_target_count"] <= effective["datasets"]["counts_per_bucket"]


def test_phase2_orchestrator_full_stage_writes_stub_outputs_for_gpu_phases(tmp_path):
    proc, out = _run(tmp_path, "full", scaffold_gpu_stages=True)
    assert proc.returncode == 0, proc.stderr
    loc = json.loads((out / "phase2_localization.json").read_text())
    inter = json.loads((out / "phase2_interventions.json").read_text())
    cot = json.loads((out / "phase2_cot_recruitment_compare.json").read_text())
    gate = json.loads((out / "phase2_gate_summary.json").read_text())
    assert loc["schema_version"] == "operator_localization_v1"
    assert loc["status"] == "not_implemented"
    assert inter["schema_version"] == "operator_intervention_sweep_v1"
    assert inter["status"] == "not_implemented"
    assert cot["schema_version"] == "cot_recruitment_compare_v1"
    assert cot["status"] == "not_implemented"
    assert gate["run_metadata"]["preregistration_used"] == "preregistration_used.json"
    assert gate["run_metadata"]["power_analysis_report"] == "power_analysis_report.json"
    assert gate["phases"]["localization_validity_gate"]["ran"] is True
    assert gate["phases"]["intervention_sanity_gate"]["ran"] is True
    assert gate["required_gates_policy"]["cot_required_for_readiness"] is True
    assert gate["scope_warnings"]
    assert gate["overall"]["ready_for_multimodel_next_tranche"] is False


def test_phase2_orchestrator_cross_operator_verify_stage_writes_verify_artifact_in_scaffold_mode(tmp_path):
    proc, out = _run(tmp_path, "cross_operator_verify", scaffold_gpu_stages=True, config=FULL_CONFIG)
    assert proc.returncode == 0, proc.stderr
    verify = json.loads((out / "phase2_cross_operator_verify.json").read_text())
    gate = json.loads((out / "phase2_gate_summary.json").read_text())
    assert verify["schema_version"] == "phase2_cross_operator_verify_v1"
    assert verify["status"] == "not_implemented"
    assert gate["phases"]["operator_specificity_gate"]["output"] == "phase2_cross_operator_verify.json"


def test_phase2_orchestrator_full_stage_scope_block_present_for_addition_only(tmp_path):
    proc, out = _run(tmp_path, "full", scaffold_gpu_stages=True, config=CONFIG)
    assert proc.returncode == 0, proc.stderr
    gate = json.loads((out / "phase2_gate_summary.json").read_text())
    assert "single_operator_scope_blocks_specificity" in gate["scope_blocks"]


def test_phase2_orchestrator_full_stage_no_single_operator_scope_block_for_full_config(tmp_path):
    proc, out = _run(tmp_path, "full", scaffold_gpu_stages=True, config=FULL_CONFIG)
    assert proc.returncode == 0, proc.stderr
    gate = json.loads((out / "phase2_gate_summary.json").read_text())
    assert "single_operator_scope_blocks_specificity" not in gate["scope_blocks"]


def test_phase2_orchestrator_operator_filter_limits_dataset_scope(tmp_path):
    proc, out = _run(
        tmp_path,
        "datasets",
        config=FULL_CONFIG,
        extra_args=["--operators", "addition"],
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out / "dataset_manifest.json").read_text())
    run_manifest = json.loads((out / "run_manifest.json").read_text())
    assert sorted(manifest["counts_by_operator"].keys()) == ["addition"]
    assert run_manifest["operator_scope"] == "subset"
    assert run_manifest["operator_coverage"] == ["addition"]


def test_phase2_orchestrator_operator_shard_mode_sets_merge_scope_block(tmp_path):
    proc, out = _run(
        tmp_path,
        "datasets",
        config=FULL_CONFIG,
        extra_args=["--operators", "addition", "--operator-shard-mode"],
    )
    assert proc.returncode == 0, proc.stderr
    gate = json.loads((out / "phase2_gate_summary.json").read_text())
    assert gate["scope"]["is_sharded_run"] is True
    assert gate["scope"]["merge_required_for_full_claims"] is True
    assert "operator_shard_requires_merge" in gate["scope_blocks"]


def test_phase2_orchestrator_strict_split_failure_hard_stops_pre_gpu(tmp_path):
    split_fail_cfg = tmp_path / "split_fail.yaml"
    cfg_text = CONFIG.read_text()
    cfg_text = cfg_text.replace("counts_per_bucket: 64", "counts_per_bucket: 1")
    split_fail_cfg.write_text(cfg_text)

    proc, out = _run(tmp_path, "full", scaffold_gpu_stages=False, config=split_fail_cfg)
    assert proc.returncode == 0, proc.stderr
    gate = json.loads((out / "phase2_gate_summary.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert gate["overall"]["phase2_status"] == "blocked_pre_gpu_split_failure"
    assert "selection_eval_split_not_applied" in gate["scope_blocks"]
    assert any(
        reason == "scope_block:selection_eval_split_not_applied"
        for reason in gate["overall"]["readiness_block_reasons"]
    )
    assert manifest["effective_config"]["datasets"]["target_operator_selection_eval_split"]["hard_stop_on_failure"] is True
    assert not (out / "phase2_localization.json").exists()


def test_selection_eval_split_separates_target_operator_prompts():
    suite = generate_operator_bucket_suite(
        {"addition": ["no_carry", "single_carry"], "subtraction": ["no_borrow"]},
        counts_per_bucket=20,
        seed=3,
        representation_variants=["plain"],
    )
    split = _split_target_operator_datasets_for_selection_eval(
        suite,
        operator="addition",
        holdout_fraction=0.5,
        min_examples_per_split=6,
        seed=17,
    )
    selection = split["selection_target_datasets"]
    eval_target = split["evaluation_target_datasets"]
    eval_all = split["evaluation_datasets_all_operators"]

    for name, sel_ds in selection.items():
        eval_ds = eval_target[name]
        assert sel_ds.operator == "addition"
        assert eval_ds.operator == "addition"
        sel_prompts = {row.prompt for row in sel_ds.examples}
        eval_prompts = {row.prompt for row in eval_ds.examples}
        assert sel_prompts.isdisjoint(eval_prompts)
        assert len(sel_prompts) > 0
        assert len(eval_prompts) > 0

    assert any(ds.operator == "subtraction" for ds in eval_all.values())

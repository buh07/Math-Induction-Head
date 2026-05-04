import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "phase2" / "run_operator_bottleneck_suite.py"
MERGE_SCRIPT = ROOT / "scripts" / "phase2" / "merge_operator_shards.py"
FULL_CONFIG = ROOT / "configs" / "phase2" / "operator_buckets_llama3_full_operators.yaml"


def _run_shard(tmp_path: Path, operator: str) -> Path:
    out = tmp_path / f"shard_{operator}"
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--stage",
        "full",
        "--smoke",
        "--scaffold-gpu-stages",
        "--dataset-config",
        str(FULL_CONFIG),
        "--operators",
        operator,
        "--operator-shard-mode",
        "--output-root",
        str(out),
        "--low-cpu-mode",
        "--max-cpu-threads",
        "1",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return out


def test_merge_operator_shards_merges_scope_and_writes_manifests(tmp_path):
    shard_add = _run_shard(tmp_path, "addition")
    shard_sub = _run_shard(tmp_path, "subtraction")
    merged = tmp_path / "merged"
    cmd = [
        sys.executable,
        str(MERGE_SCRIPT),
        "--shard-dirs",
        str(shard_add),
        str(shard_sub),
        "--output-root",
        str(merged),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    run_manifest = json.loads((merged / "run_manifest.json").read_text())
    gate = json.loads((merged / "phase2_gate_summary.json").read_text())
    merge_manifest = json.loads((merged / "merge_manifest.json").read_text())

    assert run_manifest["is_operator_shard"] is False
    assert set(run_manifest["operator_coverage"]) == {"addition", "subtraction"}
    assert gate["scope"]["is_sharded_run"] is False
    assert gate["scope"]["merge_required_for_full_claims"] is False
    assert gate["phases"]["operator_specificity_gate"]["evidence_source"] == "post_merge_cross_operator_verify"
    assert gate["phases"]["operator_specificity_gate"]["passes"] is False
    assert merge_manifest["schema_version"] == "phase2_operator_shard_merge_manifest_v1"
    assert "operator_bucket_signatures" in merge_manifest


def test_merge_operator_shards_rejects_operator_bucket_signature_mismatch(tmp_path):
    shard_a = _run_shard(tmp_path, "addition")
    shard_b = _run_shard(tmp_path, "addition")

    # Corrupt one shard's declared operator bucket signature to simulate incompatible setup.
    dataset_manifest_path = shard_b / "dataset_manifest.json"
    payload = json.loads(dataset_manifest_path.read_text())
    payload["config_snapshot"]["operator_buckets"]["addition"] = ["no_carry"]
    dataset_manifest_path.write_text(json.dumps(payload, indent=2))

    merged = tmp_path / "merged_bad_sig"
    cmd = [
        sys.executable,
        str(MERGE_SCRIPT),
        "--shard-dirs",
        str(shard_a),
        str(shard_b),
        "--output-root",
        str(merged),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "operator bucket" in (proc.stderr + proc.stdout).lower()

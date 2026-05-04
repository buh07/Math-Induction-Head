import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_script_wrappers_print_deprecation_and_help():
    wrappers = [
        ROOT / "scripts" / "run_full_experiment.py",
        ROOT / "scripts" / "detect_induction_heads.py",
        ROOT / "scripts" / "run_head_validity_suite.py",
        ROOT / "scripts" / "summarize_results.py",
    ]
    for wrapper in wrappers:
        proc = subprocess.run(
            [sys.executable, str(wrapper), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (wrapper, proc.stdout, proc.stderr)
        combined = (proc.stdout + proc.stderr).lower()
        assert "deprecated" in combined


def test_phase2_cli_help_runs():
    script = ROOT / "scripts" / "phase2" / "run_operator_bottleneck_suite.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "operator heuristic bottleneck" in proc.stdout.lower()
    assert "--stage" in proc.stdout
    assert "--operators" in proc.stdout
    assert "--batch-autotune" in proc.stdout

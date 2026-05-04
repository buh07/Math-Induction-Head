#!/usr/bin/env python3
from __future__ import annotations

import copy
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path('/scratch2/f004ndc/Math Induction Head')
CONFIG_PATH = ROOT / 'configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml'
CURRENT_RUN_ROOT = ROOT / 'results/phase2/campaign_20260305_025436_gpu01'
CURRENT_LOG_ROOT = ROOT / 'logs/phase2/campaign_20260305_025436_gpu01'

CONT_RUN_ROOT = Path(os.environ['CONT_RUN_ROOT'])
CONT_LOG_ROOT = Path(os.environ['CONT_LOG_ROOT'])

PHASE2_MODELS: Dict[str, Dict[str, Any]] = {
    'llama3_8b': {'hf_model': 'meta-llama/Meta-Llama-3-8B', 'batch_size': 8},
    'gemma_2b': {'hf_model': 'google/gemma-2b', 'batch_size': 12},
    'gpt2': {'hf_model': 'gpt2', 'batch_size': 24},
}

TARGET_UNITS = [
    'phase2:llama3_8b',
    'phase2:gemma_2b',
    'phase2:gpt2',
    'phase1:llama3_8b_rerun',
]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_cmd(cmd: List[str], *, env: Optional[Dict[str, str]] = None, log_file: Optional[Path] = None) -> int:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    if log_file is None:
        proc = subprocess.run(cmd, cwd=ROOT, env=merged_env)
        return int(proc.returncode)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open('a', encoding='utf-8') as f:
        f.write(f"\n[{dt.datetime.now().isoformat()}] CMD: {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, cwd=ROOT, env=merged_env, stdout=f, stderr=subprocess.STDOUT)
        f.write(f"[{dt.datetime.now().isoformat()}] EXIT_CODE={proc.returncode}\n")
    return int(proc.returncode)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding='utf-8')


def _normalize_cfg_for_compare(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    runtime = out.get('runtime', {})
    if isinstance(runtime, dict):
        for key in ('devices', 'batch_size', 'operator_filter', 'operator_shard_mode', 'batch_autotune'):
            runtime.pop(key, None)
    return out


def _load_target_effective_cfg(model_hf: str) -> Dict[str, Any]:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))
    raw = copy.deepcopy(raw)
    raw.setdefault('model', {})
    raw['model']['name'] = model_hf
    from scripts.phase2.run_operator_bottleneck_suite import _normalize_phase2_config
    norm = _normalize_phase2_config(raw, smoke=False)
    return _normalize_cfg_for_compare(norm)


def _phase2_required_outputs(run_dir: Path) -> Tuple[bool, List[str]]:
    required = [
        'run_manifest.json',
        'phase2_localization.json',
        'phase2_interventions.json',
        'phase2_cot_recruitment_compare.json',
        'phase2_gate_summary.json',
    ]
    missing = [x for x in required if not (run_dir / x).exists()]
    return (len(missing) == 0), missing


def _phase1_required_outputs(run_dir: Path) -> Tuple[bool, List[str]]:
    required = ['run_manifest.json', 'phase3_gate_summary.json', 'gate_summary.json']
    missing = [x for x in required if not (run_dir / x).exists()]
    return (len(missing) == 0), missing


def _phase2_reuse_check(run_dir: Path, model_alias: str) -> Tuple[bool, str]:
    info = PHASE2_MODELS[model_alias]
    complete, missing = _phase2_required_outputs(run_dir)
    if not complete:
        return False, f"missing_outputs:{','.join(missing)}"
    try:
        gate = load_json(run_dir / 'phase2_gate_summary.json')
        manifest = load_json(run_dir / 'run_manifest.json')
    except Exception as exc:
        return False, f"artifact_parse_error:{exc}"

    status = str(gate.get('overall', {}).get('phase2_status', ''))
    if status != 'full_pipeline_complete':
        return False, f"phase2_status_not_full:{status}"

    model_name = str((manifest.get('effective_config') or {}).get('model', {}).get('name', ''))
    if model_name != info['hf_model']:
        return False, f"model_mismatch:{model_name}"

    scope = gate.get('scope', {}) if isinstance(gate.get('scope'), dict) else {}
    coverage = sorted(scope.get('operator_coverage', []) or [])
    if coverage and coverage != ['addition', 'multiplication', 'subtraction']:
        return False, f"operator_scope_not_full:{coverage}"

    target_cfg = _load_target_effective_cfg(info['hf_model'])
    run_cfg = _normalize_cfg_for_compare((manifest.get('effective_config') or {}))
    if run_cfg != target_cfg:
        return False, 'effective_config_not_comparable'

    return True, 'eligible_reuse'


def _find_pid_for_output_root(output_root: Path) -> Optional[int]:
    cmd = ['bash', '-lc', f"pgrep -f 'run_operator_bottleneck_suite.py.*--output-root {re.escape(str(output_root))}' | head -n 1"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    txt = proc.stdout.strip().splitlines()
    if not txt:
        return None
    try:
        return int(txt[0].strip())
    except Exception:
        return None


def _is_pid_alive(pid: int) -> bool:
    return Path(f'/proc/{pid}').exists()


def _collect_inventory() -> Dict[str, Any]:
    units: Dict[str, Any] = {}

    for model_alias in PHASE2_MODELS:
        key = f'phase2:{model_alias}'
        run_dir = CURRENT_RUN_ROOT / model_alias
        complete, missing = _phase2_required_outputs(run_dir) if run_dir.exists() else (False, ['run_dir_missing'])
        pid = _find_pid_for_output_root(run_dir) if run_dir.exists() else None
        if pid and _is_pid_alive(pid):
            status = 'in_progress'
            reuse_eligible = False
            reason = f'pid_alive:{pid}'
        elif run_dir.exists() and complete:
            ok, why = _phase2_reuse_check(run_dir, model_alias)
            status = 'completed' if ok else 'failed'
            reuse_eligible = bool(ok)
            reason = why
        elif run_dir.exists() and not complete:
            status = 'failed'
            reuse_eligible = False
            reason = f"incomplete_outputs:{','.join(missing)}"
        else:
            status = 'missing'
            reuse_eligible = False
            reason = 'run_dir_missing'

        units[key] = {
            'source_run_dir': str(run_dir),
            'status': status,
            'has_required_outputs': bool(complete),
            'reuse_eligible': bool(reuse_eligible),
            'reason': reason,
        }

    phase1_key = 'phase1:llama3_8b_rerun'
    phase1_dir = CURRENT_RUN_ROOT / 'phase1_rerun'
    complete, missing = _phase1_required_outputs(phase1_dir) if phase1_dir.exists() else (False, ['run_dir_missing'])
    if phase1_dir.exists() and complete:
        status = 'completed'
        reuse_eligible = True
        reason = 'eligible_reuse'
    elif phase1_dir.exists() and not complete:
        status = 'failed'
        reuse_eligible = False
        reason = f"incomplete_outputs:{','.join(missing)}"
    else:
        status = 'missing'
        reuse_eligible = False
        reason = 'run_dir_missing'
    units[phase1_key] = {
        'source_run_dir': str(phase1_dir),
        'status': status,
        'has_required_outputs': bool(complete),
        'reuse_eligible': bool(reuse_eligible),
        'reason': reason,
    }

    return {
        'schema_version': 'campaign_continuity_inventory_v1',
        'generated_at_utc': now_utc(),
        'targets': TARGET_UNITS,
        'units': units,
    }


def _wait_for_inflight_phase2_models() -> None:
    log_path = CONT_LOG_ROOT / 'continuity_runner.log'
    while True:
        inv = _collect_inventory()
        write_json(CONT_RUN_ROOT / 'continuity_inventory.json', inv)
        in_progress = [k for k, v in inv['units'].items() if v['status'] == 'in_progress']
        with log_path.open('a', encoding='utf-8') as f:
            f.write(f"[{dt.datetime.now().isoformat()}] in_progress={in_progress}\n")
        if not in_progress:
            break
        time.sleep(120)


def _parse_session_name(stdout: str) -> Optional[str]:
    for line in stdout.splitlines():
        if line.startswith('Launched tmux session:'):
            return line.split(':', 1)[1].strip()
    return None


def _launch_and_wait_phase2_shards(model_alias: str) -> Path:
    info = PHASE2_MODELS[model_alias]
    model_root = CONT_RUN_ROOT / model_alias
    shard_root = model_root / 'shards'
    log_root = CONT_LOG_ROOT / f'{model_alias}_shards'
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            'SESSION': f"phase2_cont_{model_alias}_{int(time.time())}",
            'MODEL': info['hf_model'],
            'DATASET_CONFIG': 'configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml',
            'GPU_LIST': '0,1',
            'OPERATORS': 'addition,subtraction,multiplication',
            'BATCH_SIZE': str(info['batch_size']),
            'RUN_ROOT': str(shard_root),
            'LOG_ROOT': str(log_root),
            'PYTHON_BIN': '.venv/bin/python',
        }
    )

    proc = subprocess.run(
        ['bash', 'scripts/phase2/launch_operator_shards_tmux.sh'],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    (log_root / 'launcher.stdout.log').write_text(proc.stdout + "\n" + proc.stderr, encoding='utf-8')
    if proc.returncode != 0:
        raise RuntimeError(f"Shard launcher failed for {model_alias}: {proc.returncode}")

    session = _parse_session_name(proc.stdout) or env['SESSION']
    with (log_root / 'launcher.stdout.log').open('a', encoding='utf-8') as f:
        f.write(f"session={session}\n")

    operators = ['addition', 'subtraction', 'multiplication']
    status_files = [log_root / f'{op}.status' for op in operators]

    while True:
        done = [p for p in status_files if p.exists()]
        if len(done) == len(status_files):
            break
        time.sleep(60)

    failed_ops: List[str] = []
    for op in operators:
        st = log_root / f'{op}.status'
        text = st.read_text(encoding='utf-8', errors='ignore')
        m = re.search(r'EXIT_CODE\s*=\s*(\d+)', text)
        code = int(m.group(1)) if m else 1
        if code != 0:
            failed_ops.append(op)

    if failed_ops:
        retry_log = log_root / 'failed_shard_retries.log'
        gpus = ['0', '1']
        for idx, op in enumerate(failed_ops):
            gpu = gpus[idx % len(gpus)]
            out_dir = shard_root / op
            cmd = [
                '.venv/bin/python',
                'scripts/phase2/run_operator_bottleneck_suite.py',
                '--model', info['hf_model'],
                '--dataset-config', 'configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml',
                '--devices', gpu,
                '--stage', 'full',
                '--batch-size', str(info['batch_size']),
                '--operators', op,
                '--operator-shard-mode',
                '--batch-autotune',
                '--batch-autotune-stages', 'localize,intervene,cot',
                '--batch-equivalence-check',
                '--low-cpu-mode',
                '--max-cpu-threads', '2',
                '--output-root', str(out_dir),
            ]
            rc = run_cmd(cmd, env={'CUDA_VISIBLE_DEVICES': gpu, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}, log_file=retry_log)
            (log_root / f'{op}.status').write_text(f'EXIT_CODE={rc}\n', encoding='utf-8')
            if rc != 0:
                raise RuntimeError(f"Failed shard retry still failing: model={model_alias} op={op}")

    merge_out = model_root / 'merged'
    cmd = [
        '.venv/bin/python',
        'scripts/phase2/merge_operator_shards.py',
        '--shard-dirs',
        str(shard_root / 'addition'),
        str(shard_root / 'subtraction'),
        str(shard_root / 'multiplication'),
        '--output-root',
        str(merge_out),
    ]
    rc = run_cmd(cmd, log_file=log_root / 'merge.log')
    if rc != 0:
        raise RuntimeError(f"merge failed for {model_alias}")

    return merge_out


def _run_phase1_rerun() -> Path:
    out_dir = CONT_RUN_ROOT / 'phase1_rerun'
    log_file = CONT_LOG_ROOT / 'phase1_rerun.log'
    cmd = [
        '.venv/bin/python',
        'scripts/phase1/run_head_validity_suite.py',
        '--model', 'meta-llama/Meta-Llama-3-8B',
        '--devices', '0',
        '--batch-size', '8',
        '--seed-list', '0,1',
        '--phase3-primary-interventions', 'ablation,amplification',
        '--phase3-primary-k-values', '5,10',
        '--phase3-primary-scales', '0.0,1.25,1.5,2.0',
        '--phase3-multiplicity', 'bh_fdr',
        '--phase3-q-max', '0.10',
        '--phase3-require-complete-primary-coverage',
        '--output-root', str(out_dir),
    ]
    env = {
        'CUDA_VISIBLE_DEVICES': '0',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'TOKENIZERS_PARALLELISM': 'false',
    }
    rc = run_cmd(cmd, env=env, log_file=log_file)
    if rc != 0:
        cmd_retry = cmd.copy()
        idx = cmd_retry.index('--batch-size')
        cmd_retry[idx + 1] = '4'
        rc = run_cmd(cmd_retry, env=env, log_file=log_file)
    (CONT_LOG_ROOT / 'phase1_rerun.status').write_text(f'EXIT_CODE={rc}\n', encoding='utf-8')
    if rc != 0:
        raise RuntimeError('Phase1 rerun failed after retry')
    return out_dir


def main() -> int:
    CONT_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CONT_LOG_ROOT.mkdir(parents=True, exist_ok=True)

    write_json(
        CONT_RUN_ROOT / 'continuity_context.json',
        {
            'schema_version': 'campaign_continuity_context_v1',
            'generated_at_utc': now_utc(),
            'current_run_root': str(CURRENT_RUN_ROOT),
            'current_log_root': str(CURRENT_LOG_ROOT),
            'phase2_config': str(CONFIG_PATH),
            'targets': TARGET_UNITS,
        },
    )

    _wait_for_inflight_phase2_models()

    inv = _collect_inventory()
    write_json(CONT_RUN_ROOT / 'continuity_inventory.json', inv)

    canonical_phase2: Dict[str, Dict[str, Any]] = {}

    for model_alias in PHASE2_MODELS:
        unit_key = f'phase2:{model_alias}'
        unit = inv['units'][unit_key]
        if unit.get('reuse_eligible'):
            canonical_phase2[model_alias] = {
                'status': 'completed',
                'canonical_path': unit['source_run_dir'],
                'source_type': 'reused_full_run',
                'source_paths': [unit['source_run_dir']],
                'reason': unit.get('reason'),
            }
            continue

        merged = _launch_and_wait_phase2_shards(model_alias)
        canonical_phase2[model_alias] = {
            'status': 'completed',
            'canonical_path': str(merged),
            'source_type': 'sharded_merged_rerun',
            'source_paths': [
                str(CONT_RUN_ROOT / model_alias / 'shards' / 'addition'),
                str(CONT_RUN_ROOT / model_alias / 'shards' / 'subtraction'),
                str(CONT_RUN_ROOT / model_alias / 'shards' / 'multiplication'),
            ],
            'reason': 'rerun_from_scratch_sharded',
        }

    phase1_unit = inv['units']['phase1:llama3_8b_rerun']
    if phase1_unit.get('reuse_eligible'):
        canonical_phase1 = {
            'status': 'completed',
            'canonical_path': phase1_unit['source_run_dir'],
            'source_type': 'reused_full_run',
            'source_paths': [phase1_unit['source_run_dir']],
            'reason': phase1_unit.get('reason'),
        }
    else:
        out = _run_phase1_rerun()
        canonical_phase1 = {
            'status': 'completed',
            'canonical_path': str(out),
            'source_type': 'rerun_from_scratch',
            'source_paths': [str(out)],
            'reason': 'missing_or_incomplete_prior_phase1',
        }

    campaign_complete = all(v.get('status') == 'completed' for v in canonical_phase2.values()) and canonical_phase1.get('status') == 'completed'

    continuity_manifest = {
        'schema_version': 'campaign_continuity_manifest_v1',
        'generated_at_utc': now_utc(),
        'phase2_models': canonical_phase2,
        'phase1_rerun': canonical_phase1,
        'campaign_complete': bool(campaign_complete),
        'notes': [
            'Legacy artifacts were not mutated.',
            'Mid-stage resume was not used; reruns were unit-level from scratch.',
        ],
    }
    write_json(CONT_RUN_ROOT / 'campaign_continuity_manifest.json', continuity_manifest)

    final_inv = _collect_inventory()
    write_json(CONT_RUN_ROOT / 'continuity_inventory.json', final_inv)
    return 0


if __name__ == '__main__':
    try:
        rc = main()
    except Exception as exc:
        err = {
            'schema_version': 'campaign_continuity_error_v1',
            'generated_at_utc': now_utc(),
            'error_type': exc.__class__.__name__,
            'error': str(exc),
        }
        write_json(CONT_RUN_ROOT / 'campaign_continuity_error.json', err)
        raise
    sys.exit(rc)

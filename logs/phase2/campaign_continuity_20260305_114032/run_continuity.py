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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding='utf-8')


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _log(msg: str) -> None:
    p = CONT_LOG_ROOT / 'continuity_runner.log'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as f:
        f.write(f"[{dt.datetime.now().isoformat()}] {msg}\n")


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
    # Match exact output-root token in cmdline where possible.
    cmd = ['bash', '-lc', f"pgrep -f 'run_operator_bottleneck_suite.py.*--output-root {re.escape(str(output_root))}' | head -n 1"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    rows = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
    if not rows:
        return None
    try:
        return int(rows[0])
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
    units['phase1:llama3_8b_rerun'] = {
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
    while True:
        inv = _collect_inventory()
        write_json(CONT_RUN_ROOT / 'continuity_inventory.json', inv)
        in_progress = [k for k, v in inv['units'].items() if v['status'] == 'in_progress']
        _log(f'in_progress={in_progress}')
        if not in_progress:
            return
        time.sleep(120)


def _launch_op_process(model_alias: str, operator: str, gpu: str, out_dir: Path, log_file: Path) -> subprocess.Popen:
    info = PHASE2_MODELS[model_alias]
    env = os.environ.copy()
    env.update(
        {
            'CUDA_VISIBLE_DEVICES': gpu,
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'TOKENIZERS_PARALLELISM': 'false',
        }
    )
    cmd = [
        '.venv/bin/python',
        'scripts/phase2/run_operator_bottleneck_suite.py',
        '--model', info['hf_model'],
        '--dataset-config', 'configs/phase2/operator_buckets_llama3_full_operators_campaign.yaml',
        '--devices', gpu,
        '--stage', 'full',
        '--batch-size', str(info['batch_size']),
        '--operators', operator,
        '--operator-shard-mode',
        '--batch-autotune',
        '--batch-autotune-stages', 'localize,intervene,cot',
        '--batch-equivalence-check',
        '--low-cpu-mode',
        '--max-cpu-threads', '2',
        '--output-root', str(out_dir),
    ]
    log_file.parent.mkdir(parents=True, exist_ok=True)
    f = log_file.open('w', encoding='utf-8')
    f.write(f"[{dt.datetime.now().isoformat()}] START operator={operator} gpu={gpu} model={info['hf_model']}\n")
    f.flush()
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
    proc._log_handle = f  # type: ignore[attr-defined]
    return proc


def _close_proc_log(proc: subprocess.Popen) -> None:
    h = getattr(proc, '_log_handle', None)
    if h is not None:
        try:
            h.write(f"[{dt.datetime.now().isoformat()}] EXIT_CODE={proc.returncode}\n")
            h.flush()
            h.close()
        except Exception:
            pass


def _run_phase2_model_sharded(model_alias: str) -> Path:
    model_root = CONT_RUN_ROOT / model_alias
    shard_root = model_root / 'shards'
    log_root = CONT_LOG_ROOT / f'{model_alias}_shards'
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    operators = ['addition', 'subtraction', 'multiplication']
    gpu_slots = ['0', '1']
    queue = operators.copy()
    running: Dict[str, Tuple[subprocess.Popen, str, str]] = {}
    status: Dict[str, int] = {}

    # Start up to two operators in parallel.
    while queue and len(running) < len(gpu_slots):
        op = queue.pop(0)
        gpu = gpu_slots[len(running) % len(gpu_slots)]
        proc = _launch_op_process(model_alias, op, gpu, shard_root / op, log_root / f'{op}.log')
        running[op] = (proc, gpu, op)
        _log(f'shard_started model={model_alias} op={op} gpu={gpu} pid={proc.pid}')

    while running:
        finished_ops: List[str] = []
        for op, (proc, gpu, _op) in running.items():
            rc = proc.poll()
            if rc is None:
                continue
            _close_proc_log(proc)
            status[op] = int(rc)
            (log_root / f'{op}.status').write_text(f'EXIT_CODE={rc}\n', encoding='utf-8')
            _log(f'shard_finished model={model_alias} op={op} gpu={gpu} rc={rc}')
            finished_ops.append(op)

        for op in finished_ops:
            running.pop(op, None)

        while queue and len(running) < len(gpu_slots):
            op = queue.pop(0)
            used = {gpu for (_p, gpu, _o) in running.values()}
            gpu = next((g for g in gpu_slots if g not in used), gpu_slots[0])
            proc = _launch_op_process(model_alias, op, gpu, shard_root / op, log_root / f'{op}.log')
            running[op] = (proc, gpu, op)
            _log(f'shard_started model={model_alias} op={op} gpu={gpu} pid={proc.pid}')

        if running:
            time.sleep(30)

    failed = [op for op, rc in status.items() if rc != 0]
    if failed:
        _log(f'shard_failures model={model_alias} failed_ops={failed}; retrying once serially')
        for idx, op in enumerate(failed):
            gpu = gpu_slots[idx % len(gpu_slots)]
            proc = _launch_op_process(model_alias, op, gpu, shard_root / op, log_root / f'{op}.retry1.log')
            rc = proc.wait()
            _close_proc_log(proc)
            (log_root / f'{op}.status').write_text(f'EXIT_CODE={rc}\n', encoding='utf-8')
            _log(f'shard_retry_finished model={model_alias} op={op} rc={rc}')
            if rc != 0:
                raise RuntimeError(f'shard retry failed: model={model_alias} op={op}')

    merge_out = model_root / 'merged'
    merge_cmd = [
        '.venv/bin/python',
        'scripts/phase2/merge_operator_shards.py',
        '--shard-dirs',
        str(shard_root / 'addition'),
        str(shard_root / 'subtraction'),
        str(shard_root / 'multiplication'),
        '--output-root', str(merge_out),
    ]
    merge_log = log_root / 'merge.log'
    with merge_log.open('w', encoding='utf-8') as f:
        f.write(f"[{dt.datetime.now().isoformat()}] CMD: {' '.join(merge_cmd)}\n")
        proc = subprocess.run(merge_cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"[{dt.datetime.now().isoformat()}] EXIT_CODE={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f'merge failed for {model_alias}')

    # sanity check merged outputs
    complete, missing = _phase2_required_outputs(merge_out)
    if not complete:
        raise RuntimeError(f'merged output incomplete for {model_alias}: {missing}')

    return merge_out


def _run_phase1_rerun() -> Path:
    out_dir = CONT_RUN_ROOT / 'phase1_rerun'
    log_file = CONT_LOG_ROOT / 'phase1_rerun.log'

    base_cmd = [
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
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})

    def _run(cmd: List[str]) -> int:
        with log_file.open('a', encoding='utf-8') as f:
            f.write(f"\n[{dt.datetime.now().isoformat()}] CMD: {' '.join(cmd)}\n")
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
            f.write(f"[{dt.datetime.now().isoformat()}] EXIT_CODE={proc.returncode}\n")
        return int(proc.returncode)

    rc = _run(base_cmd)
    if rc != 0:
        retry = base_cmd.copy()
        i = retry.index('--batch-size')
        retry[i + 1] = '4'
        rc = _run(retry)

    (CONT_LOG_ROOT / 'phase1_rerun.status').write_text(f'EXIT_CODE={rc}\n', encoding='utf-8')
    if rc != 0:
        raise RuntimeError('phase1 rerun failed after one retry')

    complete, missing = _phase1_required_outputs(out_dir)
    if not complete:
        raise RuntimeError(f'phase1 output incomplete: {missing}')
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
            'policy': {
                'run_policy': 'finish_then_shard_rest',
                'reuse_policy': 'keep_if_scientifically_equivalent',
            },
        },
    )

    _wait_for_inflight_phase2_models()

    inv = _collect_inventory()
    write_json(CONT_RUN_ROOT / 'continuity_inventory.json', inv)

    canonical_phase2: Dict[str, Dict[str, Any]] = {}
    for model_alias in PHASE2_MODELS:
        key = f'phase2:{model_alias}'
        unit = inv['units'][key]
        if unit.get('reuse_eligible'):
            canonical_phase2[model_alias] = {
                'status': 'completed',
                'canonical_path': unit['source_run_dir'],
                'source_type': 'reused_full_run',
                'source_paths': [unit['source_run_dir']],
                'reason': unit.get('reason', ''),
            }
            _log(f'reused phase2 unit model={model_alias} path={unit["source_run_dir"]}')
            continue

        merged_out = _run_phase2_model_sharded(model_alias)
        canonical_phase2[model_alias] = {
            'status': 'completed',
            'canonical_path': str(merged_out),
            'source_type': 'sharded_merged_rerun',
            'source_paths': [
                str(CONT_RUN_ROOT / model_alias / 'shards' / 'addition'),
                str(CONT_RUN_ROOT / model_alias / 'shards' / 'subtraction'),
                str(CONT_RUN_ROOT / model_alias / 'shards' / 'multiplication'),
            ],
            'reason': 'rerun_from_scratch_sharded',
        }
        _log(f'rerun-complete phase2 model={model_alias} merged={merged_out}')

    phase1_unit = inv['units']['phase1:llama3_8b_rerun']
    if phase1_unit.get('reuse_eligible'):
        canonical_phase1 = {
            'status': 'completed',
            'canonical_path': phase1_unit['source_run_dir'],
            'source_type': 'reused_full_run',
            'source_paths': [phase1_unit['source_run_dir']],
            'reason': phase1_unit.get('reason', ''),
        }
        _log(f'reused phase1 unit path={phase1_unit["source_run_dir"]}')
    else:
        out_dir = _run_phase1_rerun()
        canonical_phase1 = {
            'status': 'completed',
            'canonical_path': str(out_dir),
            'source_type': 'rerun_from_scratch',
            'source_paths': [str(out_dir)],
            'reason': 'missing_or_incomplete_prior_phase1',
        }
        _log(f'rerun-complete phase1 path={out_dir}')

    campaign_complete = all(v.get('status') == 'completed' for v in canonical_phase2.values()) and canonical_phase1.get('status') == 'completed'

    write_json(
        CONT_RUN_ROOT / 'campaign_continuity_manifest.json',
        {
            'schema_version': 'campaign_continuity_manifest_v1',
            'generated_at_utc': now_utc(),
            'phase2_models': canonical_phase2,
            'phase1_rerun': canonical_phase1,
            'campaign_complete': bool(campaign_complete),
            'notes': [
                'Legacy artifacts were not mutated.',
                'No mid-stage resume was used; reruns were unit-level from scratch.',
            ],
        },
    )

    write_json(CONT_RUN_ROOT / 'continuity_inventory.json', _collect_inventory())
    _log(f'campaign_complete={campaign_complete}')
    return 0


if __name__ == '__main__':
    try:
        rc = main()
    except Exception as exc:
        write_json(
            CONT_RUN_ROOT / 'campaign_continuity_error.json',
            {
                'schema_version': 'campaign_continuity_error_v1',
                'generated_at_utc': now_utc(),
                'error_type': exc.__class__.__name__,
                'error': str(exc),
            },
        )
        raise
    sys.exit(rc)

"""Run E0 to a soft six-hour/ten-step boundary, then match E1/E2/E3.

Each arm starts from the same SFT. Refuse occupied GPUs, existing runs or
insufficient checkpoint rotation capacity; never kill an unrelated Ray job.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

import requests

from prepare_formal50 import CODE_ROOT, R, W as PREFLIGHT, resolved
from tau3_grpo.experiments.manifest import read_manifest
from tau3_grpo.training.services import launch_process, stop_process

W = Path(os.environ['TAU3_RUN_ROOT']) / 'rl-c50-matched6h-a800-20260912'
owned = []


def write_state(**state):
    state['updated_at'] = time.time()
    temporary = W / 'controller-state.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(W / 'controller-state.json')
    print(json.dumps(state), flush=True)


def stage_config(arm, updates, started_at=None, resume_from=None):
    from tau3_grpo.tracking.swanlab import load_tracking_env
    os.environ.setdefault('TAU3_ENV_FILE', str(R / 'code/.env'))
    load_tracking_env()
    command, env, snapshot = resolved(arm, total_updates=updates)
    assert 'fa2-overlay' not in env.get('PYTHONPATH', ''), 'Only the validated FLA overlay is allowed'
    assert env['VERL_QWEN35_FLA_IEEE'] == '1'
    assert env['VERL_QWEN35_TRIM_PADDING'] == 'experimental_both'
    assert env['TRITON_F32_DEFAULT'] == 'ieee'
    env.pop('RAY_ADDRESS', None)
    env.update(TRAINER_LOGGERS='[console,swanlab]', SWANLAB_MODE='online', GIT_CONFIG_COUNT='1',
               GIT_CONFIG_KEY_0='safe.directory', GIT_CONFIG_VALUE_0=str(CODE_ROOT))
    # Explicit runtime propagation: TaskRunner must see the budget and arm.
    runtime = {'TAU3_GRPO_ARM': arm, 'TAU3_SWANLAB_CONTINUITY': '1',
               'TAU3_KEEP_COMPLETE_BOUNDARY': '1',
               'TAU3_ENV_FILE': env['TAU3_ENV_FILE'], 'SWANLAB_MODE': 'online',
               'SWANLAB_LOG_DIR': env['SWANLAB_LOG_DIR'],
               'TAU3_STOP_REQUEST_PATH': str(W / 'STOP_AFTER_BOUNDARY'),
               'TAU3_BUDGET_STATE_PATH': str(W / f'{arm}_seed42/budget.json'),
               'TAU3_BUDGET_INTERVAL': '10'}
    if arm == 'e0' and resume_from is None:
        runtime.update(TAU3_E0_DISCOVERY_SECONDS='21600',
                       TAU3_BUDGET_STARTED_AT=str(started_at if started_at is not None else time.time()),
                       TAU3_BUDGET_STATE_PATH=str(W / 'e0_seed42/budget.json'),
                       TAU3_BUDGET_INTERVAL='10')
    if resume_from is not None:
        resume_from = Path(resume_from).resolve()
        restored = int(resume_from.name.removeprefix('global_step_'))
        if restored % 10 or updates % 10 or restored >= updates:
            raise ValueError('Resume from a completed ten-step boundary to a later ten-step target')
        command += ['trainer.resume_mode=resume_path', f'trainer.resume_from_path={resume_from}']
        # Resume from the exact existing schedule; its audited prefix may be
        # longer than the new stopping target (E0 discovery prepared 100).
        env['TRAIN_PARQUET'] = str(resume_from.parent / 'train_schedule.parquet')
    env.update(runtime)
    command += [f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}='{value}'" for key, value in runtime.items()]
    command += ['++ray_kwargs.ray_init.address=local']
    snapshot.update(command=command, matched_budget_runtime=runtime,
                    trainer_loggers=['console', 'swanlab'], actual_updates_or_discovery_ceiling=updates)
    return command, env, snapshot


def launch(command, env, name):
    process = launch_process(command, cwd=CODE_ROOT, env=env,
                             log_path=W / f'{name}.log', pid_path=W / f'{name}.pid')
    owned.append(process)
    return process


def stop(process):
    stop_process(process)


def active_pids(devices):
    output = subprocess.check_output(['nvidia-smi', '-i', devices,
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return {int(line) for line in output.splitlines() if line.strip()}


def verify_result(arm, target):
    result = W / f'{arm}_seed42'
    latest = int((result / 'latest_checkpointed_iteration.txt').read_text())
    assert latest == target, f'{arm}: checkpoint {latest} != matched target {target}'
    actor = result / f'global_step_{target}/actor'
    for prefix in ('model', 'optim', 'extra_state'):
        shards = list(actor.glob(f'{prefix}_world_size_4_rank_*.pt'))
        assert len(shards) == 4 and all(p.stat().st_size > 0 for p in shards), (arm, prefix)
    telemetry = [json.loads(line) for line in (result / 'telemetry.jsonl').read_text().splitlines()]
    assert len(telemetry) == target
    evaluations = {}
    for step in range(10, target + 1, 10):
        rows = [json.loads(line) for line in (result / f'validation/{step}.jsonl').read_text().splitlines()]
        assert len(rows) == 240, f'{arm}/{step}: selection60 x 4 missing rows'
        evaluations[str(step)] = {'trajectories': len(rows),
                                 'mean_score': sum(row['score'] for row in rows) / len(rows)}
    summary = {'arm': arm, 'target_step': target, 'evaluations': evaluations,
               'latest_complete_checkpoint': str(actor), 'fresh_sft_start': True}
    import swanlab
    tracking = json.loads((result / 'swanlab-run.json').read_text())
    for attempt in range(6):
        remote = swanlab.Api().run(tracking['run_path'])
        metrics = remote.metrics(keys=['trainer/global_step'], all=True)
        points = [point for row in metrics.get('list', []) for point in row.get('metrics', [])]
        if sorted((int(p['step']), int(p['value'])) for p in points) == [(s, s) for s in range(1, target + 1)]:
            break
        time.sleep(2)
    else:
        raise RuntimeError('Saved boundary exists but SwanLab step readback is incomplete; inspect before continuation')
    summary['swanlab'] = {'run_id': tracking['run_id'], 'url': tracking['url'],
                         'cloud_steps_verified': target}
    (result / 'completion.json').write_text(json.dumps(summary, indent=2))
    return summary


def main(dry_run):
    from tau3_grpo.tracking.swanlab import load_tracking_env
    load_tracking_env()
    W.mkdir(parents=True, exist_ok=True)
    assert (PREFLIGHT / 'preflight.json').is_file(), 'Run prepare_formal50.py first'
    if dry_run:
        for arm in ('e0', 'e1', 'e2', 'e3'):
            command, env, snapshot = stage_config(arm, 100 if arm == 'e0' else 30)
            dry = dict(env, TAU3_DRY_RUN='1')
            output = subprocess.check_output(command, env=dry, cwd=CODE_ROOT, text=True)
            assert 'enable_thinking=false' in output and 'tool_execution_mode=sequential' in output
            (PREFLIGHT / f'{arm}.matched.command.txt').write_text(output)
            # Invoke Hydra directly from the dry shell expansion; do not run
            # launcher GPU checks or materialize a formal schedule here.
            import shlex
            argv = shlex.split(output.splitlines()[-1]) + ['--cfg', 'job']
            hydra = subprocess.check_output(argv, env=env, cwd=CODE_ROOT, text=True)
            (PREFLIGHT / f'{arm}.matched.hydra.yaml').write_text(hydra)
            (PREFLIGHT / f'{arm}.matched.launch.json').write_text(json.dumps(snapshot, indent=2))
        print('Four Hydra configurations resolved; no service or training started.', flush=True)
        return

    if (W / 'HOLD_UNTIL_USER_START').exists():
        write_state(status='waiting_for_user_start', no_training_launched=True)
        return 0

    for arm in ('e0', 'e1', 'e2', 'e3'):
        assert not (W / f'{arm}_seed42').exists(), 'Existing run: inspect before any manual recovery'
    assert not active_pids('0,1,2,3,4'), 'GPU occupied; no services started'
    assert shutil.disk_usage(W).free > 110 * 2**30, 'Need space for old + new full checkpoint and margin'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 8100))
    (W / 'controller.started').open('x').close()
    started = time.time()
    (W / 'gpu-assignment.txt').write_text(subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv'], text=True)
        + '\npolicy=0,1,2,3; simulator=4\n')
    # Whole-device samples complement allocator metrics for long trajectories.
    # This observer belongs to this controller and exits with its other children.
    launch(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits', '--loop=15'], os.environ.copy(), 'gpu-memory')
    _, simenv, _ = stage_config('e0', 100, started)
    simenv.update(TAU3_USER_CUDA_DEVICES='4', TAU3_USER_MAX_NUM_SEQS='16',
                  TAU3_USER_MAX_MODEL_LEN='16384', TAU3_USER_GPU_MEMORY_UTILIZATION='.65',
                  TAU3_USER_ENFORCE_EAGER='1', TAU3_USER_PORT='8100')
    write_state(status='starting_simulator', e0_started_at=started, requested_seconds=21600)
    simulator = launch(['bash', str(CODE_ROOT / 'scripts/serve/simulator_qwen38.sh')], simenv, 'simulator')
    deadline = time.monotonic() + 1200
    while True:
        if simulator.poll() is not None:
            raise RuntimeError('Simulator exited during startup')
        try:
            if requests.get('http://127.0.0.1:8100/health', timeout=3).status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('Simulator startup timeout')
        time.sleep(3)

    completed = []
    target = None
    for arm in ('e0', 'e1', 'e2', 'e3'):
        # Keep the previous durable checkpoint until the new one finishes.
        # Never use tmpfs as the only recovery point or delete another arm.
        free = shutil.disk_usage(W).free / 2**30
        if free <= 110:
            write_state(status='waiting_for_storage', next_arm=arm, target_step=target,
                        completed=completed, available_gib=free, required_free_gib=110)
            return 2
        assert not active_pids('0,1,2,3'), 'Policy GPU occupied; refusing to launch next arm'
        updates = 100 if arm == 'e0' else target
        if (W / 'STOP_AFTER_BOUNDARY').exists():
            write_state(status='paused_at_boundary', next_arm=arm, target_step=target, completed=completed)
            return 0
        command, env, snapshot = stage_config(arm, updates, started)
        result = Path(env['RESULTS_DIR'])
        # Fix a shared schedule before training, and compare actual prefixes.
        from tau3_grpo.experiments.prepare import prepare_experiment_inputs
        from prepare_formal50 import CANDIDATES
        prepare_experiment_inputs(arm=arm, seed=42, data_seed=42, group_size=8,
            groups_per_update=8, total_updates=updates, anchor_mode='structured',
            manifest_dir=CANDIDATES / 'manifests', output_dir=result)
        if arm != 'e0':
            expected = read_manifest(W / 'e0_seed42').schedule[:target]
            assert read_manifest(result).schedule == expected
        (result / 'launch.json').write_text(json.dumps(snapshot, indent=2))
        write_state(status='training', active_arm=arm, target_step=target,
                    discovery_ceiling=100, e0_started_at=started, completed=completed)
        train = launch(command, env, arm)
        rc = train.wait()
        (W / f'{arm}.exit').write_text(str(rc))
        if rc:
            raise RuntimeError(f'{arm} exited {rc}; later arms not started')
        budget = json.loads((result / 'budget.json').read_text())
        if budget.get('stop_requested'):
            completed.append(verify_result(arm, budget['target_step']))
            write_state(status='paused_at_boundary', active_arm=arm,
                        completed_step=budget['target_step'], target_step=target, completed=completed)
            return 0
        if arm == 'e0':
            target = budget['target_step']
            assert target and 0 < target <= 100 and target % 10 == 0
        completed.append(verify_result(arm, target))
        deadline = time.monotonic() + 180
        while active_pids('0,1,2,3') and time.monotonic() < deadline:
            time.sleep(3)
    write_state(status='complete', target_step=target, completed=completed)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Controller interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        rc = main(args.dry_run) or 0
        if not args.dry_run:
            (W / 'controller.exit').write_text(str(rc))
    except BaseException as exc:
        if not args.dry_run:
            write_state(status='failed', error=f'{type(exc).__name__}: {exc}')
            (W / 'controller.exit').write_text('1')
        raise
    finally:
        for child in reversed(owned):
            stop(child)

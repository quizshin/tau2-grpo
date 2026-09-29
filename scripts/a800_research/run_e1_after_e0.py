"""Continue E1 on persistent storage, then E2/E3 on user-authorized tmpfs."""
import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import requests

import run_matched50 as controller
from prepare_formal50 import CANDIDATES, CODE_ROOT
from tau3_grpo.experiments.manifest import read_manifest
from tau3_grpo.experiments.prepare import prepare_experiment_inputs


def prepare_run(arm, target):
    result = controller.W / f'{arm}_seed42'
    assert not result.exists() and not result.is_symlink(), 'Existing run must not be overwritten'
    if arm in ('e2', 'e3'):
        temporary = Path('/dev/shm/tau3-matched50-20260913') / result.name
        assert not temporary.exists()
        assert shutil.disk_usage('/dev/shm').free > 110 * 2**30
        # tmpfs pages count against the same cgroup RAM limit as training.
        cgroup = Path('/sys/fs/cgroup/memory')
        limit = int((cgroup / 'memory.limit_in_bytes').read_text())
        usage = int((cgroup / 'memory.usage_in_bytes').read_text())
        assert limit - usage > 160 * 2**30, 'Insufficient RAM headroom for tmpfs checkpoint rotation'
        temporary.mkdir(parents=True)
        result.symlink_to(temporary, target_is_directory=True)
    else:
        assert shutil.disk_usage(controller.W).free > 110 * 2**30
    command, env, snapshot = controller.stage_config(arm, target)
    prepare_experiment_inputs(arm=arm, seed=42, data_seed=42, group_size=8,
        groups_per_update=8, total_updates=target, anchor_mode='structured',
        manifest_dir=CANDIDATES / 'manifests', output_dir=result)
    assert read_manifest(result).schedule == read_manifest(controller.W / 'e0_seed42').schedule[:target]
    snapshot['storage'] = {'path': str(result.resolve()), 'temporary': arm in ('e2', 'e3')}
    (result / 'launch.json').write_text(json.dumps(snapshot, indent=2))
    return command, env, result


def persist_tmpfs_models(arm, step):
    """Keep final model shards and experiment evidence durable; retain full tmpfs checkpoint."""
    import hashlib
    source = controller.W / f'{arm}_seed42'
    destination = controller.W / 'persistent-models' / source.name
    checkpoint = source / f'global_step_{step}'
    weight_files = list((checkpoint / 'actor').glob('model_world_size_4_rank_*.pt'))
    assert len(weight_files) == 4
    assert shutil.disk_usage(controller.W).free > sum(p.stat().st_size for p in weight_files) + 5 * 2**30
    assert not destination.exists()
    staging = destination.with_name(destination.name + '.copying')
    staging.mkdir(parents=True)
    hashes = {}
    # Keep all small artifacts, rollouts and evaluations; optimizer remains in tmpfs.
    for item in source.iterdir():
        if item.name.startswith('global_step_'):
            continue
        if item.is_dir():
            shutil.copytree(item, staging / item.name, symlinks=True)
        elif item.is_file():
            shutil.copy2(item, staging / item.name)
    out = staging / f'global_step_{step}' / 'actor'
    out.mkdir(parents=True)
    for item in (checkpoint / 'actor').iterdir():
        if item.name.startswith('optim_world_size_'):
            continue
        if item.is_dir():
            shutil.copytree(item, out / item.name, symlinks=True)
        elif item.is_file():
            shutil.copy2(item, out / item.name)
    for item in checkpoint.iterdir():
        if item.is_file() and item.name != 'checkpoint-complete.json':
            shutil.copy2(item, out.parent / item.name)
    for item in weight_files:
        pair = []
        for path in (item, out / item.name):
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    h.update(block)
            pair.append(h.hexdigest())
        assert pair[0] == pair[1]
        hashes[item.name] = pair[0]
    (staging / 'archive.json').write_text(json.dumps({'arm': arm, 'step': step,
        'model_sha256': hashes, 'includes_optimizer': False,
        'full_resume_checkpoint_in_tmpfs': str(checkpoint.resolve()),
        'warning': 'Exact optimizer continuation needs tmpfs checkpoint; export before shutdown.'}, indent=2))
    staging.rename(destination)
    return str(destination)


def completed_e0():
    root = controller.W / 'e0_seed42'
    result = json.loads((root / 'completion.json').read_text())
    target = result['target_step']
    assert result['arm'] == 'e0' and target > 0 and target % 10 == 0
    assert (controller.W / 'e0.exit').read_text().strip() == '0'
    assert int((root / 'latest_checkpointed_iteration.txt').read_text()) == target
    checkpoint = root / f'global_step_{target}'
    receipt = json.loads((checkpoint / 'checkpoint-complete.json').read_text())
    assert receipt['step'] == target and receipt['world_size'] == 4
    for name, size in receipt['files'].items():
        assert (checkpoint / name).stat().st_size == size, name
    assert result['swanlab']['cloud_steps_verified'] == target
    for step in range(10, target + 1, 10):
        assert len((root / f'validation/{step}.jsonl').read_text().splitlines()) == 240
    return target, result


def main(dry_run=False):
    target, e0 = completed_e0()
    command, env, snapshot = controller.stage_config('e1', target)
    assert 'TAU3_E0_DISCOVERY_SECONDS' not in env
    if dry_run:
        output = subprocess.check_output(command, cwd=CODE_ROOT,
            env=dict(env, TAU3_DRY_RUN='1'), text=True)
        argv = shlex.split(output.splitlines()[-1]) + ['--cfg', 'job']
        config = subprocess.check_output(argv, cwd=CODE_ROOT, env=env, text=True)
        (controller.PREFLIGHT / 'e1.u20.continuation.hydra.yaml').write_text(config)
        print(json.dumps({'dry_run_passed': True, 'arm': 'e1', 'target': target}), flush=True)
        return 0

    w = controller.W
    previous = json.loads((w / 'controller-state.json').read_text())
    assert previous['status'] == 'waiting_for_storage' and previous['next_arm'] == 'e1'
    assert previous['target_step'] == target
    assert all(not (w / f'{arm}_seed42').exists() for arm in ('e1', 'e2', 'e3'))
    assert not (w / 'STOP_AFTER_BOUNDARY').exists()
    assert not (w / 'HOLD_UNTIL_USER_START').exists()
    assert not controller.active_pids('0,1,2,3,4'), 'GPU occupied'
    assert shutil.disk_usage(w).free > 110 * 2**30
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 8100))
    (w / 'e1-controller.started').open('x').close()
    archive = w / 'controller-before-e1-20260913'
    archive.mkdir(exist_ok=True)
    shutil.copy2(w / 'controller-state.json', archive / 'controller-state.json')
    if (w / 'controller.exit').exists():
        (w / 'controller.exit').rename(archive / 'controller.exit')

    command, env, result = prepare_run('e1', target)
    controller.launch(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits', '--loop=15'], os.environ.copy(), 'gpu-memory-e1')
    simenv = dict(env, TAU3_USER_CUDA_DEVICES='4', TAU3_USER_MAX_NUM_SEQS='16',
        TAU3_USER_MAX_MODEL_LEN='16384', TAU3_USER_GPU_MEMORY_UTILIZATION='.65',
        TAU3_USER_ENFORCE_EAGER='1', TAU3_USER_PORT='8100')
    controller.write_state(status='starting_simulator', active_arm='e1', target_step=target,
                           completed=[e0], e1_started_at=time.time())
    simulator = controller.launch(['bash', str(CODE_ROOT / 'scripts/serve/simulator_qwen38.sh')],
                                  simenv, 'simulator-e1')
    deadline = time.monotonic() + 1200
    while True:
        if simulator.poll() is not None:
            raise RuntimeError('E1 simulator exited during startup')
        try:
            if requests.get('http://127.0.0.1:8100/health', timeout=3).status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('E1 simulator startup timeout')
        time.sleep(3)
    completed = [e0]
    for arm in ('e1', 'e2', 'e3'):
        if (w / 'STOP_AFTER_BOUNDARY').exists():
            controller.write_state(status='paused_at_boundary', next_arm=arm,
                                   target_step=target, completed=completed)
            return 0
        deadline = time.monotonic() + 180
        while controller.active_pids('0,1,2,3') and time.monotonic() < deadline:
            time.sleep(3)
        assert not controller.active_pids('0,1,2,3')
        if arm != 'e1':
            command, env, result = prepare_run(arm, target)
        controller.write_state(status='training', active_arm=arm, target_step=target,
            completed=completed, checkpoint_storage=str(result.resolve()),
            temporary_checkpoint=arm in ('e2', 'e3'))
        train = controller.launch(command, env, arm)
        rc = train.wait()
        (w / f'{arm}.exit').write_text(str(rc))
        if rc:
            raise RuntimeError(f'{arm} exited {rc}')
        budget = json.loads((result / 'budget.json').read_text())
        stopped = budget.get('stop_requested', False)
        done = controller.verify_result(arm, budget['target_step'] if stopped else target)
        if arm in ('e2', 'e3'):
            controller.write_state(status='persisting_model', active_arm=arm,
                                   target_step=target, completed=completed)
            done['persistent_model_archive'] = persist_tmpfs_models(arm, done['target_step'])
            done['full_checkpoint_temporary'] = True
        completed.append(done)
        if stopped:
            controller.write_state(status='paused_at_boundary', active_arm=arm,
                                   target_step=target, completed=completed)
            return 0
    controller.write_state(status='complete', target_step=target, completed=completed,
        full_optimizer_checkpoints_for_e2_e3_are_temporary=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt('E1 controller interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        rc = main(args.dry_run)
        if not args.dry_run:
            (controller.W / 'controller.exit').write_text(str(rc))
    except BaseException as exc:
        if not args.dry_run:
            controller.write_state(status='failed', error=f'{type(exc).__name__}: {exc}')
            (controller.W / 'controller.exit').write_text('1')
        raise
    finally:
        for child in reversed(controller.owned):
            controller.stop(child)

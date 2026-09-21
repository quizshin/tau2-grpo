"""Run one matched GRPO, GiGPO or MT-GTPO experiment using the existing training/serving paths.

Activate the canonical remote environment first. --dry-run checks data, source,
shell expansion and Hydra without starting a service. Resume requires a complete
ten-step boundary in the same result directory and the original SwanLab identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import tarfile
import time
from pathlib import Path
from string import Template

import requests
import yaml

from tau3_grpo.configuration import load_config
from tau3_grpo.data.manifest import read_manifest as read_data_manifest
from tau3_grpo.experiments.manifest import read_manifest
from tau3_grpo.experiments.prepare import prepare_experiment_inputs
from tau3_grpo.launch import prepare
from tau3_grpo.paths import CODE_ROOT, DATA_ROOT
from tau3_grpo.tracking.rl_continuity import atomic_json
from tau3_grpo.tracking.swanlab import load_tracking_env, redact_config
from tau3_grpo.training.rl.checkpoints import validate_checkpoint
from tau3_grpo.training.services import launch_process, stop_process

PROFILES = {
    version: CODE_ROOT / f'configs/train/rl/formal50_mt_gtpo_{version}.yaml'
    for version in ('v2', 'v3', 'paper_v1', 'paper_env_v2', 'paper_env_split_v3', 'paper_env_split_v4')
}
BASE_PROFILE = CODE_ROOT / 'configs/train/rl/formal50_a800.yaml'
PAPER_VERSIONS = {'paper_v1', 'paper_env_v2', 'paper_env_split_v3', 'paper_env_split_v4'}
MANIFEST_SHA = '641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae'


def resolve(result, *, updates=20, dynamic_filter=False, resume_from=None, reward_version='v3',
            reward_recipe=None, allow_uncalibrated=False, estimator='mt_gtpo'):
    from tau3_grpo.evaluation.process_reward import reward_settings
    from tau3_grpo.evaluation.rewards.recipe import load_frozen_recipe
    from tau3_grpo.integrations.verl.mt_gtpo import settings_from_config

    if estimator not in {'grpo', 'tau_gigpo', 'mt_gtpo'}:
        raise ValueError(f'Unknown estimator: {estimator}')
    arm = ('mt_gtpo' if estimator == 'mt_gtpo' else
           ('e3' if dynamic_filter else 'e2') if estimator == 'tau_gigpo' else
           ('e1' if dynamic_filter else 'e0'))
    if estimator != 'mt_gtpo' and reward_recipe is not None:
        raise ValueError('A process reward recipe requires mt_gtpo')
    paper_run = estimator == 'mt_gtpo' and reward_version in PAPER_VERSIONS
    profile = PROFILES[reward_version] if estimator == 'mt_gtpo' else BASE_PROFILE
    frozen = None
    expected_reward = reward_settings({
        'mode': 'paper' if reward_version in PAPER_VERSIONS else 'reference_write',
        'version': reward_version,
    })
    if reward_recipe is not None:
        if reward_version not in PAPER_VERSIONS:
            raise ValueError('Frozen IRC recipes require --reward-version paper_v1, paper_env_v2, paper_env_split_v3 or paper_env_split_v4')
        frozen = load_frozen_recipe(reward_recipe, manifest_sha256=MANIFEST_SHA)
        if frozen['reward']['version'] != reward_version:
            raise ValueError('Frozen recipe reward version differs from requested version')
        expected_reward = frozen['reward']
    elif paper_run and (not allow_uncalibrated or resume_from):
        raise ValueError('Formal paper run requires a passed --reward-recipe; initializer is dry-run only')
    if updates < 10 or updates % 10:
        raise ValueError('Formal targets must be positive multiples of ten')
    result = Path(result).resolve()
    reward_label = reward_version if reward_version in PAPER_VERSIONS else f'refwrite-{reward_version}'
    if estimator != 'mt_gtpo':
        reward_label = 'outcome'
    env = dict(os.environ, CODE_ROOT=str(CODE_ROOT))
    for key in ('TRAIN_PARQUET', 'ROLLOUT_DATA_DIR', 'RAY_ADDRESS', 'TAU3_E0_DISCOVERY_SECONDS',
                'TAU3_BUDGET_STARTED_AT', 'TAU3_DRY_RUN'):
        env.pop(key, None)
    # Explicit profile settings override activation defaults, as in the E0 controller.
    profile_environment = load_config(profile)['launch']['environment']
    for key, value in profile_environment.items():
        env[key] = Template(str(value)).substitute(env)
    controller_environment = dict(RESULTS_DIR=str(result), TOOL_CONFIG=str(result / 'tool_config.yaml'),
               SWANLAB_LOG_DIR=str(result / 'swanlog'), TOTAL_UPDATES=str(updates),
               TAU3_GRPO_DEBUG_BATCH_DIR=str(result / 'update-batches'),
               VERL_QWEN35_WEIGHT_AUDIT_DIR=str(result / 'weight-audits'),
               SWANLAB_EXPERIMENT_NAME=f'{estimator.upper()}-Qwen35-4B-full-c50-g8x8-u{updates}-{reward_label}-df-{int(dynamic_filter)}-s42')
    env.update(controller_environment)
    runtime = {'TAU3_RECORD_TRAJECTORY_FACTS': '1', 'TAU3_GRPO_ARM': arm, 'TAU3_SWANLAB_CONTINUITY': '1',
               'TAU3_KEEP_COMPLETE_BOUNDARY': '1', 'SWANLAB_MODE': 'online',
               'TAU3_ENV_FILE': env.get('TAU3_ENV_FILE', str(CODE_ROOT / '.env')),
               'SWANLAB_LOG_DIR': env['SWANLAB_LOG_DIR'],
               'TAU3_STOP_REQUEST_PATH': str(result / 'STOP_AFTER_BOUNDARY'),
               'TAU3_BUDGET_STATE_PATH': str(result / 'budget.json'), 'TAU3_BUDGET_INTERVAL': '10',
               'TAU3_GRPO_DEBUG_BATCH_DIR': env['TAU3_GRPO_DEBUG_BATCH_DIR']}
    env.update(runtime)
    extra = [f'algorithm.dynamic_filter.enable={str(dynamic_filter).lower()}',
             'actor_rollout_ref.rollout.calculate_log_probs=true',
             'actor_rollout_ref.rollout.logprobs_mode=processed_logprobs',
             '++ray_kwargs.ray_init.address=local']
    if frozen:
        for section in ('weights', 'paper_options'):
            extra += [f'++algorithm.process_reward.{section}.{key}={value}'
                      for key, value in frozen['reward'][section].items()]
        extra += [f'++algorithm.mt_gtpo.{key}={value}' for key, value in frozen['algorithm'].items()]
    extra += [f'++ray_kwargs.ray_init.runtime_env.env_vars.{k}={json.dumps(v)}' for k, v in runtime.items()]
    if resume_from:
        previous = yaml.safe_load((result / 'resolved-hydra.yaml').read_text())
        previous_algorithm = previous.get('algorithm', {})
        if (previous_algorithm.get('adv_estimator') != estimator
                or (estimator == 'mt_gtpo' and previous_algorithm.get('process_reward', {}).get('version') != reward_version)
                or previous_algorithm.get('dynamic_filter', {}).get('enable') != dynamic_filter):
            raise ValueError('Resume cannot change estimator, reward version or dynamic filtering; '
                             'missing identity metadata must be recovered before resuming')
        if frozen:
            old_recipe = load_frozen_recipe(result / 'frozen-recipe.json', manifest_sha256=MANIFEST_SHA)
            if (old_recipe['sha256'] != frozen['sha256']
                    or reward_settings(previous['algorithm']['process_reward']) != expected_reward
                    or settings_from_config(previous['algorithm']) != frozen['algorithm']):
                raise ValueError('Resume cannot change frozen IRC recipe or Hybrid settings')
        checkpoint = Path(resume_from).resolve()
        step = int(checkpoint.name.removeprefix('global_step_'))
        if checkpoint.parent != result or step % 10 or not 0 < step < updates:
            raise ValueError('Resume must use this run and an earlier completed ten-step checkpoint')
        validate_checkpoint(result, step)
        if not (result / 'swanlab-run.json').is_file():
            raise ValueError('Missing SwanLab identity')
        extra += ['trainer.resume_mode=resume_path', f'trainer.resume_from_path={checkpoint}']
    command, env, snapshot = prepare('rl', profile, arm, 42, extra, env)
    snapshot['environment_sources'] = {
        key: ('formal_controller' if key in controller_environment or key in runtime else
              'formal_profile' if key in profile_environment else source)
        for key, source in snapshot['environment_sources'].items()
    }
    snapshot['configuration_precedence'] = (
        'inherited environment < explicit formal profile < controller arguments; '
        'native Hydra overrides appended last'
    )
    snapshot.update(command=command, formal_runtime=runtime, dynamic_filter=dynamic_filter,
                    total_updates=updates, result_dir=str(result), estimator=estimator)
    if paper_run:
        snapshot['irc_recipe'] = frozen
        snapshot['uncalibrated_initializer'] = frozen is None
    return command, env, redact_config(snapshot)


def validate_inputs(env, result, updates, reward_version='v3', estimator='mt_gtpo'):
    import pandas as pd
    source = Path(env['TRAIN_MANIFEST_DIR']) / 'areal_airline_train_seed42.jsonl'
    if hashlib.sha256(source.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise ValueError('Training pool differs from the historical 50-task comparison')
    train = read_data_manifest(source)
    selection = read_data_manifest(DATA_ROOT / 'manifests/areal_airline_selection_seed42.jsonl')
    assert len(train) == 50 and len(selection) == 60
    # Check reference coverage; this is not measured rollout reward density.
    from tau3_grpo.envs.tau2_bridge import task_model
    from tau3_grpo.evaluation.process_reward import DB_WRITE_TOOLS
    tasks = [task_model().model_validate(json.loads(line)['task'])
             for line in source.read_text().splitlines() if line.strip()]
    eligible = (sum(bool(task.evaluation_criteria.actions) for task in tasks)
                if reward_version in PAPER_VERSIONS else
                sum(any(item.value == 'DB' for item in task.evaluation_criteria.reward_basis)
                   and any(action.name in DB_WRITE_TOOLS for action in task.evaluation_criteria.actions or [])
                   for task in tasks))
    if estimator == 'mt_gtpo' and not eligible:
        raise ValueError('reference_write has no positive process reward on this task pool; '
                         'review reward semantics before a formal training run')
    assert not ({x.task_id for x in train} & {x.task_id for x in selection})
    for entry in train + selection:
        db = DATA_ROOT / 'raw/areal_tau2' / entry.db_path
        assert hashlib.sha256(db.read_bytes()).hexdigest() == entry.db_hash
    actual = pd.read_parquet(env['VAL_PARQUET'])
    assert [x['task_id'] for x in actual.extra_info] == [x.task_id for x in selection]
    for path in (env['MODEL_PATH'], env['TAU3_USER_MODEL']):
        assert Path(path).is_dir() and list(Path(path).glob('*.safetensors')), path
    prepare_experiment_inputs(arm=env['TAU3_GRPO_ARM'], seed=42, data_seed=42, group_size=8,
        groups_per_update=8, total_updates=updates, anchor_mode='structured',
        manifest_dir=source.parent, output_dir=result)
    historical = Path(env['TAU3_RUN_ROOT']) / 'rl-c50-matched6h-a800-20260912/e0_seed42'
    if (historical / 'experiment_manifest.json').is_file():
        baseline = read_manifest(historical).schedule
        current = read_manifest(result).schedule
        length = min(len(baseline), len(current))
        assert current[:length] == baseline[:length], 'Historical task schedule differs'
    return {'train_tasks': 50, 'selection_tasks': 60, 'manifest_sha256': MANIFEST_SHA,
            'positive_process_reward_eligible_tasks': eligible if estimator == 'mt_gtpo' else None,
            'estimator': estimator,
            'reward_version': reward_version if estimator == 'mt_gtpo' else None,
            'reward_scope': ('official outcome only' if estimator != 'mt_gtpo' else
                             'environment-adapted split gold-read/gold-DB-write tiers; frozen IRC weights'
                             if reward_version in {'paper_env_split_v3', 'paper_env_split_v4'} else
                             'environment-adapted paper tiers; execution inputs and response-aware read duplicates'
                             if reward_version == 'paper_env_v2' else
                             'paper-derived tiers, frozen IRC weights, no DB-write budget'
                             if reward_version == 'paper_v1' else
                             'ungated execution-equivalent DB reference writes; positive budget 1; errors -0.1'
                             if reward_version == 'v3' else
                             'success-gated exact DB reference writes; positive budget 1; errors -0.1'),
            'updates': updates, 'candidate_trajectories': updates * 64,
            'historical_schedule_checked': (historical / 'experiment_manifest.json').is_file()}


def snapshot_source(destination):
    destination.mkdir()
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                                    cwd=CODE_ROOT).decode().split('\0')
    hashes = {}
    with tarfile.open(destination / 'source.tar.gz', 'w:gz') as archive:
        for name in sorted(set(names) - {''}):
            path = CODE_ROOT / name
            if (name.startswith(('.env', 'data/', 'models/', 'checkpoints/', 'results/'))
                    or not path.is_file() or path.is_symlink()):
                continue
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name, recursive=False)
    atomic_json(destination / 'source-sha256.json', hashes)
    for args, filename in [(['rev-parse', 'HEAD'], 'git-head.txt'),
                           (['status', '--short'], 'git-status.txt'),
                           (['diff', '--binary', 'HEAD', '--', '.', ':!.env', ':!results', ':!data'], 'worktree.patch')]:
        (destination / filename).write_bytes(subprocess.check_output(['git', *args], cwd=CODE_ROOT))


def active_gpu_pids():
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    return [int(x) for x in raw.splitlines() if x.strip()]


def verify_completion(result, target):
    latest = int((result / 'latest_checkpointed_iteration.txt').read_text())
    budget = json.loads((result / 'budget.json').read_text())
    expected = budget['target_step'] if budget.get('stop_requested') else target
    assert latest == expected
    validate_checkpoint(result, latest)
    scores = {}
    for step in range(10, latest + 1, 10):
        rows = [json.loads(x) for x in (result / f'validation/{step}.jsonl').read_text().splitlines()]
        assert len(rows) == 240
        scores[step] = sum(x['score'] for x in rows) / 240
    import swanlab
    tracking = json.loads((result / 'swanlab-run.json').read_text())
    for _ in range(6):
        values = swanlab.Api().run(tracking['run_path']).metrics(keys=['trainer/global_step'], all=True)
        points = [(int(p['step']), int(p['value'])) for row in values.get('list', []) for p in row.get('metrics', [])]
        if sorted(points) == [(s, s) for s in range(1, latest + 1)]:
            break
        time.sleep(2)
    else:
        raise RuntimeError('Cloud step readback incomplete; checkpoint retained')
    return {'completed_step': latest, 'evaluation_scores': scores, 'swanlab_url': tracking['url'],
            'cloud_steps_verified': latest, 'paused': bool(budget.get('stop_requested'))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--estimator', choices=['grpo', 'tau_gigpo', 'mt_gtpo'], default='mt_gtpo')
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--dynamic-filter', action='store_true')
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--reward-version', choices=sorted(PROFILES), default='v3')
    parser.add_argument('--reward-recipe', type=Path, help='Passed frozen IRC recipe matching the paper reward version')
    args = parser.parse_args(argv)
    load_tracking_env()
    result = args.result_dir.resolve()
    command, env, snapshot = resolve(result, updates=args.updates,
        dynamic_filter=args.dynamic_filter, resume_from=args.resume_from, reward_version=args.reward_version,
        reward_recipe=args.reward_recipe, allow_uncalibrated=args.dry_run, estimator=args.estimator)
    if (result / 'STOP_AFTER_BOUNDARY').exists():
        raise ValueError('Stop request exists; inspect and archive it before resuming')
    if not args.resume_from and ((result / 'swanlab-run.json').exists() or list(result.glob('global_step_*'))):
        raise ValueError('Existing training state: use explicit resume, never restart over this run')
    if not args.dry_run and (result / 'controller-state.json').is_file() and not args.resume_from:
        raise ValueError('Existing controller attempt: inspect and use a fresh result directory')
    result.mkdir(parents=True, exist_ok=True)
    report = validate_inputs(env, result, args.updates, args.reward_version, args.estimator)
    dry = subprocess.check_output(command, env=dict(env, TAU3_DRY_RUN='1'), cwd=CODE_ROOT, text=True)
    (result / 'resolved-command.txt').write_text(dry)
    hydra = subprocess.check_output(shlex.split(dry.splitlines()[-1]) + ['--cfg', 'job'],
                                   env=env, cwd=CODE_ROOT, text=True)
    (result / 'resolved-hydra.yaml').write_text(hydra)
    config = yaml.safe_load(hydra)
    assert config['algorithm']['adv_estimator'] == args.estimator
    if args.estimator == 'mt_gtpo':
        assert config['algorithm']['process_reward']['mode'] == (
            'paper' if args.reward_version in PAPER_VERSIONS else 'reference_write')
        assert config['algorithm']['process_reward']['version'] == args.reward_version
    assert config['algorithm']['dynamic_filter']['enable'] == args.dynamic_filter
    assert config['trainer']['save_freq'] == config['trainer']['test_freq'] == 10
    assert config['trainer']['max_actor_ckpt_to_keep'] == 1
    assert config['trainer']['total_training_steps'] == args.updates
    if args.estimator == 'mt_gtpo' and args.reward_version in PAPER_VERSIONS:
        from tau3_grpo.evaluation.process_reward import reward_settings
        from tau3_grpo.integrations.verl.mt_gtpo import settings_from_config

        frozen = snapshot['irc_recipe']
        report['irc_calibrated'] = frozen is not None
        if frozen:
            assert reward_settings(config['algorithm']['process_reward']) == frozen['reward']
            assert settings_from_config(config['algorithm']) == frozen['algorithm']
            atomic_json(result / 'frozen-recipe.json', frozen)
    atomic_json(result / 'launch.json', snapshot)
    atomic_json(result / 'preflight.json', report)
    if args.dry_run:
        print(json.dumps(report, indent=2), flush=True)
        return
    if active_gpu_pids():
        raise RuntimeError('GPU occupied; refusing to start services')
    if shutil.disk_usage(result).free < 110 * 2**30:
        raise RuntimeError('Need 110 GiB free for safe complete-checkpoint rotation')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 8100))
    session = result / f'controller-session-{time.time_ns()}'
    session.mkdir()
    snapshot_source(session / 'source')
    children = []

    def state(status, **extra):
        atomic_json(result / 'controller-state.json', dict(status=status, pid=os.getpid(),
                    updated_at=time.time(), target=args.updates, session=str(session), **extra))

    def launch(argv, process_env, name):
        process = launch_process(argv, cwd=CODE_ROOT, env=process_env,
                                 log_path=session / f'{name}.log',
                                 pid_path=session / f'{name}.pid')
        children.append(process)
        return process

    try:
        state('starting_simulator')
        simenv = dict(env, TAU3_USER_CUDA_DEVICES='4', TAU3_USER_MAX_NUM_SEQS='16',
                      TAU3_USER_MAX_MODEL_LEN='16384', TAU3_USER_GPU_MEMORY_UTILIZATION='.65',
                      TAU3_USER_ENFORCE_EAGER='1', TAU3_USER_PORT='8100')
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
        state('training')
        train = launch(command, env, 'train')
        rc = train.wait()
        if rc:
            raise RuntimeError(f'Training exited {rc}; inspect {session}/train.log')
        state('verifying')
        completion = verify_completion(result, args.updates)
        atomic_json(result / 'completion.json', completion)
        state('paused' if completion['paused'] else 'completed', **completion)
    except BaseException as exc:
        state('failed', error=str(exc))
        raise
    finally:
        for process in reversed(children):
            stop_process(process)


if __name__ == '__main__':
    main()

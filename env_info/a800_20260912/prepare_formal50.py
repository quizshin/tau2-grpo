"""Audit and materialize the four fixed-budget runs; never launch RL/services."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from string import Template

from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.parquet_builder import build_rows, write_parquet
from tau3_grpo.envs.adapter import airline_policy
from tau3_grpo.experiments.prepare import prepare_experiment_inputs
from tau3_grpo.experiments.manifest import read_manifest as read_schedule, flatten_schedule
from tau3_grpo.launch import load_config, prepare
from tau3_grpo.paths import CODE_ROOT, DATA_ROOT


R = Path(os.environ['TAU3_ROOT'])
W = Path(os.environ['TAU3_RUN_ROOT']) / 'formal50-preflight-20260912'
CANDIDATES = CODE_ROOT / 'results/analysis/rl_curriculum50_20260912'
EXPECTED_MANIFEST_SHA = '641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae'


def resolved(arm, *, total_updates=50):
    if arm not in ('e0', 'e1', 'e2', 'e3'):
        raise ValueError(arm)
    profile = CODE_ROOT / f'configs/train/rl/qwen35_4b_full_a800_c50_matched6h_{arm}_20260912.yaml'
    env = dict(os.environ, CODE_ROOT=str(CODE_ROOT))
    env.pop('TRAIN_PARQUET', None)
    # The activated host carries older training defaults; apply this profile
    # explicitly. Generic launcher precedence remains unchanged.
    for key, value in load_config(profile)['launch']['environment'].items():
        env[key] = Template(str(value)).substitute(env)
    env['TOTAL_UPDATES'] = str(total_updates)
    env['SWANLAB_EXPERIMENT_NAME'] = f'{arm.upper()}-Qwen35-4B-full-c50-g8x8-matched6h-s42'
    for key in ('TAU3_E0_DISCOVERY_SECONDS', 'TAU3_BUDGET_STARTED_AT', 'TAU3_BUDGET_STATE_PATH', 'TAU3_BUDGET_INTERVAL'):
        env.pop(key, None)
    result = Path(env['RESULTS_DIR'])
    env.update(TOOL_CONFIG=str(result / 'tool_config.yaml'),
               SWANLAB_LOG_DIR=str(result / 'swanlog'),
               VERL_QWEN35_WEIGHT_AUDIT_DIR=str(result / 'weight-audits'))
    command, env, snapshot = prepare('rl', profile, arm, 42, [], env)
    assert int(env['GROUPS_PER_UPDATE']) * int(env['GROUP_SIZE']) == 64
    assert env['TAU3_GRPO_DEBUG_BATCH_DIR'] == ''
    assert result.name == f'{arm}_seed42'
    return command, env, snapshot


def main():
    import pandas as pd
    from collections import Counter

    W.mkdir(parents=True, exist_ok=True)
    path = CANDIDATES / 'manifests/areal_airline_train_seed42.jsonl'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_MANIFEST_SHA
    candidates = read_manifest(path)
    parent = {entry.task_id: entry for entry in read_manifest(DATA_ROOT / 'manifests/areal_airline_train_seed42.jsonl')}
    selection = read_manifest(DATA_ROOT / 'manifests/areal_airline_selection_seed42.jsonl')
    assert len(candidates) == len({entry.task_id for entry in candidates}) == 50
    assert len(selection) == len({entry.task_id for entry in selection}) == 60
    assert not ({entry.task_id for entry in candidates} & {entry.task_id for entry in selection})
    assert all(entry == parent[entry.task_id] for entry in candidates)
    for entry in candidates + selection:
        db_path = DATA_ROOT / 'raw/areal_tau2' / entry.db_path
        assert hashlib.sha256(db_path.read_bytes()).hexdigest() == entry.db_hash

    rows = build_rows(selection, policy=airline_policy(), split='selection', anchor_mode='structured', seed=42)
    val = W / 'selection60.parquet'
    if not val.exists():
        write_parquet(rows, val)
    actual = pd.read_parquet(val)
    assert [info['task_id'] for info in actual['extra_info']] == [entry.task_id for entry in selection]
    assert all(list(prompt) == expected['prompt'] for prompt, expected in zip(actual['prompt'], rows))

    schedules = []
    for arm in ('e0', 'e1', 'e2', 'e3'):
        _, env, snapshot = resolved(arm)
        prepare_experiment_inputs(arm=arm, seed=42, data_seed=42, group_size=8,
            groups_per_update=8, total_updates=50, anchor_mode='structured',
            manifest_dir=CANDIDATES / 'manifests', output_dir=W / f'plan-{arm}-u50')
        schedule = read_schedule(W / f'plan-{arm}-u50')
        counts = Counter(flatten_schedule(schedule.schedule))
        assert len(counts) == 50 and set(counts.values()) == {8}
        assert all(len(set(step.task_ids)) == 8 for step in schedule.schedule)
        schedules.append([step.to_dict() for step in schedule.schedule])
        (W / f'{arm}.launch.json').write_text(json.dumps(snapshot, indent=2))
    assert all(schedule == schedules[0] for schedule in schedules)
    report = {'kind': 'four_arm_matched_budget_preflight_no_training', 'schedule_audit_steps_only': 50, 'actual_target': 'E0 approximately 6 h then round up to 10; all arms share N', 'task_count': 50,
               'manifest_sha256': EXPECTED_MANIFEST_SHA, 'selection_task_count': 60,
               'identical_schedules': True, 'steps_per_arm': 50, 'trajectories_per_step': 64,
               'trajectories_per_arm': 3200, 'training_trajectories_all_arms': 12800,
               'evaluation_steps': [10, 20, 30, 40, 50], 'evaluation_samples_per_task': 4,
               'evaluation_trajectories_all_arms': 4800, 'save_every_steps': 10,
               'retained_complete_checkpoints_per_arm': 1,
               'latest_complete_checkpoint_gib_previous_measurement': 50.6,
               'four_latest_complete_checkpoints_gib_estimate': 202.4,
               'current_run_volume_free_gib': shutil.disk_usage(W).free / 2**30,
               'no_training_launched': True,
               'scope': 'Identity/configuration checks; online validation path and full-run time still need testing.'}
    (W / 'preflight.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

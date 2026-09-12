"""CPU-only cloud logging check: simulated step indices, never RL training."""
import json
import os
from pathlib import Path
import time

from tau3_grpo.tracking.rl_continuity import start_continuous_run
from tau3_grpo.tracking.swanlab import load_tracking_env


def main():
    load_tracking_env()
    import swanlab

    root = Path(os.environ['TAU3_RUN_ROOT']) / 'swanlab-continuity-check-20260912'
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'swanlab-run.json').exists(), 'Do not replay an existing validation run'
    config = {'stage': 'tracking-validation', 'seed': 42,
        'trainer': {'phase': 'tracking-validation', 'default_local_dir': str(root), 'total_training_steps': 30},
        'algorithm': {'validation_only': True}, 'data': {'seed': 42, 'train_batch_size': 1},
        'actor_rollout_ref': {'model': {'path': 'synthetic-no-model-loaded'}, 'rollout': {'n': 1},
                             'actor': {'optim': {'lr': 0.0}}}}
    options = {'workspace': os.environ.get('SWANLAB_WORKSPACE') or None,
               'job_type': 'test', 'group': 'tracking-validation', 'tags': ['engineering-check', 'no-RL']}
    os.environ['TAU3_RESTORED_STEP'] = '0'
    first = start_continuous_run(swanlab, project=os.environ.get('TAU3_SWANLAB_PROJECT', 'tau3-grpo-pytrio'),
        name='CHECK-only-SwanLab-resume-30-to-31-no-RL', config=config, options=options,
        log_dir=str(root / 'session1'), mode='online')
    for step in range(1, 31):
        first.log({'check/continuity': step * 2}, step)
    first.finish()
    def readback(last):
        for attempt in range(6):
            remote = swanlab.Api().run(first.state['run_path'])
            response = remote.metrics(keys=['trainer/global_step', 'check/continuity'], all=True)
            series = {row['key']: [(int(p['step']), float(p['value'])) for p in row.get('metrics', [])]
                      for row in response.get('list', [])}
            if all(len(series.get(key, [])) == last for key in ('trainer/global_step', 'check/continuity')):
                return series
            time.sleep(2)
        raise RuntimeError('Cloud metric readback incomplete')
    before = readback(30)
    os.environ['TAU3_RESTORED_STEP'] = '30'
    config['trainer']['total_training_steps'] = 50
    second = start_continuous_run(swanlab, project=first.state['project'],
        name='CHECK-only-SwanLab-resume-30-to-31-no-RL', config=config, options=options,
        log_dir=str(root / 'session2'), mode='online')
    assert second.state['run_id'] == first.state['run_id']
    second.log({'check/continuity': 62}, 31)
    second.finish()
    after = readback(31)
    for key, points in before.items():
        assert after[key][:30] == points
        assert [step for step, _ in after[key]] == list(range(1, 32))
    assert after['trainer/global_step'][-1] == (31, 31.0)
    report = {'kind': 'CPU-only SwanLab cloud continuity check; no model/GPU/RL execution',
              'run_id': second.state['run_id'], 'url': second.state['url'], 'same_run_id': True,
              'original_30_points_unchanged': True, 'next_step_is_31': True,
              'no_duplicate_or_offset_steps': True, 'cloud_series': after}
    (root / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'cloud_series'}, indent=2))


if __name__ == '__main__':
    main()

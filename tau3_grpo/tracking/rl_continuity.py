"""Persist one SwanLab identity and use actual checkpoint steps on continuation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import uuid


def atomic_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def identity(config):
    actor = config['actor_rollout_ref']
    manifest = Path(config['trainer']['default_local_dir']) / 'experiment_manifest.json'
    split_hash = json.loads(manifest.read_text()).get('train_split_hash') if manifest.is_file() else None
    value = {'algorithm': config['algorithm'], 'arm': os.environ.get('TAU3_GRPO_ARM'),
             'train_split_hash': split_hash,
             'model': actor['model']['path'], 'seed': config['data']['seed'],
             'groups': config['data']['train_batch_size'], 'samples': actor['rollout']['n'],
             'lr': actor['actor']['optim']['lr'], 'kl': actor['actor'].get('kl_loss_coef')}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_identity(config, project, workspace, checkpoint_step):
    root = Path(config['trainer']['default_local_dir'])
    path = root / 'swanlab-run.json'
    fingerprint = identity(config)
    if not path.exists() and checkpoint_step:
        checkpoint = Path(config['trainer'].get('resume_from_path') or root / f'global_step_{checkpoint_step}')
        archived = checkpoint / 'swanlab-run.json'
        if not archived.is_file():
            raise ValueError('Checkpoint has no SwanLab identity; refusing to create a different continuation curve')
        atomic_json(path, json.loads(archived.read_text()))
    if path.exists():
        state = json.loads(path.read_text())
        if state['identity'] != fingerprint or state['project'] != project:
            raise ValueError('SwanLab continuation must preserve the original arm/model/data/training settings')
        if workspace is not None and state['workspace'] != workspace:
            raise ValueError('SwanLab continuation workspace changed')
        if state.get('last_logged_step', 0) > checkpoint_step:
            raise ValueError('Curve is ahead of checkpoint; normal continuation requires a completed saved boundary')
        resume = 'must' if state.get('initialized') else 'allow'
    else:
        if checkpoint_step:
            raise ValueError('Missing SwanLab identity for checkpoint continuation')
        state = {'run_id': uuid.uuid4().hex[:21], 'project': project, 'workspace': workspace,
                 'identity': fingerprint, 'initialized': False, 'last_logged_step': 0,
                 'sessions': []}
        atomic_json(path, state)
        resume = 'allow'
    return path, state, {'id': state['run_id'], 'resume': resume,
                         'workspace': state['workspace']}


def cloud_last_step(sdk, state):
    if not state.get('run_path'):
        return 0
    remote = sdk.Api().run(state['run_path'])
    values = remote.metrics(keys=['trainer/global_step'], all=True)
    points = [point for entry in values.get('list', []) for point in entry.get('metrics', [])]
    return max((int(point['step']) for point in points), default=0)


class ContinuousSwanlab:
    def __init__(self, sdk, run, path, state, checkpoint_step):
        self.sdk, self.run, self.path, self.state = sdk, run, path, state
        self.last_step = checkpoint_step
        self.finished = False
        state.update(initialized=True, run_id=run.id, url=run.url,
                     run_path=urlparse(run.url).path.strip('/').removeprefix('@').replace('/runs/', '/'),
                     local_dir=str(run.dir), last_logged_step=checkpoint_step)
        state['sessions'].append({'restored_step': checkpoint_step, 'first_update': checkpoint_step + 1,
                                  'started_at': time.time()})
        atomic_json(path, state)

    def log(self, data, step):
        # No offsets: restoring step 30 means the next real update is 31.
        if step <= self.last_step:
            raise ValueError(f'Non-increasing SwanLab update {step}; last real step is {self.last_step}')
        if int(data.get('trainer/global_step', step)) != step:
            raise ValueError('SwanLab axis differs from the actual optimizer step')
        payload = dict(data, **{'trainer/global_step': step})
        with (self.path.parent / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps({'step': step, 'metrics': payload}, default=str) + '\n')
        self.sdk.log(data=payload, step=step)
        self.last_step = step
        self.state['last_logged_step'] = step
        atomic_json(self.path, self.state)

    def finish(self):
        if not self.finished:
            self.sdk.finish()
            self.finished = True
            self.state['sessions'][-1]['last_step'] = self.last_step
            self.state['sessions'][-1]['finished_at'] = time.time()
            atomic_json(self.path, self.state)


def start_continuous_run(sdk, *, project, name, config, options, log_dir, mode):
    if mode not in ('online', 'cloud'):
        raise ValueError('This formal run requires online SwanLab continuity')
    step = int(os.environ['TAU3_RESTORED_STEP'])
    path, state, resume = prepare_identity(config, project, options.get('workspace'), step)
    if state.get('initialized') and cloud_last_step(sdk, state) > step:
        raise ValueError('Cloud curve is ahead of the loaded checkpoint; refusing to mix repeated updates')
    run = sdk.init(project=project, name=name, config=config, log_dir=log_dir,
                   mode='online', public=False, **{**options, **resume})
    return ContinuousSwanlab(sdk, run, path, state, step)

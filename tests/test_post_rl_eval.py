import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def module():
    from tau3_grpo.evaluation import controller
    return controller


@pytest.fixture
def enough_backup_space(module, monkeypatch):
    # Unit-sized checkpoint fixtures must not depend on the host /tmp capacity.
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=100 * module.GIB))


def checkpoint(mod, root, arm='e2'):
    run = root / f'{arm}_seed42'
    cp = run / 'global_step_20'
    (cp / 'actor').mkdir(parents=True)
    files = {'data.pt': b'data'}
    files.update({f'actor/{kind}_world_size_4_rank_{rank}.pt': f'{kind}{rank}'.encode()
                  for kind in ('model', 'optim', 'extra_state') for rank in range(4)})
    for name, value in files.items():
        (cp / name).write_bytes(value)
    mod.atomic_json(cp / 'checkpoint-complete.json', {'step': 20, 'world_size': 4,
                    'files': {k: len(v) for k, v in files.items()}})
    (run / 'latest_checkpointed_iteration.txt').write_text('20')
    (root / f'{arm}.exit').write_text('0')
    mod.atomic_json(run / 'completion.json', {'arm': arm, 'target_step': 20,
                                            'swanlab': {'cloud_steps_verified': 20}})
    (run / 'validation').mkdir()
    for step in (10, 20):
        (run / f'validation/{step}.jsonl').write_text(''.join(
            json.dumps({'task_id': str(t), 'score': 0}) + '\n' for t in range(60) for _ in range(4)))
    return run


def test_complete_backup_survives_original_loss(module, tmp_path, enough_backup_space):
    run = checkpoint(module, tmp_path)
    out = tmp_path / 'post';out.mkdir()
    dest = module.persist_complete(tmp_path, out, 'e2')
    assert module.read(out / 'e2-backup.json')['includes_optimizer'] is True
    assert len(list(dest.rglob('optim_*.pt'))) == 4
    module.shutil.rmtree(run)
    module.validate_checkpoint(dest)


def test_backup_does_not_publish_corruption(module, tmp_path, monkeypatch, enough_backup_space):
    checkpoint(module, tmp_path)
    out = tmp_path / 'post';out.mkdir()
    original = module.shutil.copytree
    def corrupt(src, dest, *args, **kwargs):
        original(src, dest, *args, **kwargs)
        if (Path(dest) / 'data.pt').exists():
            (Path(dest) / 'data.pt').write_bytes(b'corruption')
    monkeypatch.setattr(module.shutil, 'copytree', corrupt)
    with pytest.raises(ValueError, match='SHA256'):
        module.persist_complete(tmp_path, out, 'e2')
    assert not (tmp_path / 'persistent-checkpoints/e2_seed42').exists()
    assert (tmp_path / 'e2_seed42/global_step_20/data.pt').read_bytes() == b'data'


def test_checkpoint_requires_optimizer_and_detects_damage(module, tmp_path):
    run = checkpoint(module, tmp_path)
    cp = module.validate_checkpoint(run)
    (cp / 'actor/optim_world_size_4_rank_2.pt').unlink()
    with pytest.raises(ValueError, match='Incomplete checkpoint'):
        module.validate_checkpoint(run)


def test_checkpoint_does_not_trust_incomplete_receipt(module, tmp_path):
    run = checkpoint(module, tmp_path)
    p = run / 'global_step_20/checkpoint-complete.json'
    d = module.read(p);d['files'].pop('data.pt');module.atomic_json(p, d)
    with pytest.raises(ValueError, match='complete recovery'):
        module.validate_checkpoint(run)


@pytest.mark.parametrize('status', ['failed', 'paused_at_boundary', 'waiting_for_storage'])
def test_failed_or_paused_rl_never_triggers_eval(module, tmp_path, status):
    module.atomic_json(tmp_path / 'controller-state.json', {'status': status})
    with pytest.raises(RuntimeError):
        module.queue_ready(tmp_path, False)


def test_success_requires_process_exit_and_all_arms(module, tmp_path):
    for arm in module.ARMS:
        checkpoint(module, tmp_path, arm)
    module.atomic_json(tmp_path / 'controller-state.json', {'status': 'complete', 'target_step': 20,
                        'completed': [{'arm': a} for a in module.ARMS]})
    assert module.queue_ready(tmp_path, False) is False
    (tmp_path / 'controller.exit').write_text('0')
    assert module.queue_ready(tmp_path, True) is False
    assert module.queue_ready(tmp_path, False) is True
    (tmp_path / 'STOP_AFTER_BOUNDARY').touch()
    with pytest.raises(RuntimeError):
        module.queue_ready(tmp_path, False)


def test_training_state_is_not_a_trigger(module, tmp_path):
    module.atomic_json(tmp_path / 'controller-state.json', {'status': 'training'})
    assert module.queue_ready(tmp_path, True) is False


def test_reused_backup_must_match_receipt(module, tmp_path, enough_backup_space):
    checkpoint(module, tmp_path)
    out = tmp_path / 'post';out.mkdir()
    dest = module.persist_complete(tmp_path, out, 'e2')
    assert module.persist_complete(tmp_path, out, 'e2') == dest
    (dest / 'global_step_20/data.pt').write_bytes(b'evil')
    with pytest.raises(ValueError, match='corrupt'):
        module.persist_complete(tmp_path, out, 'e2')


def test_summary_rejects_incomplete_or_different_tasks(module, tmp_path):
    jobs = [{'task_id': 'a', 'trial': i, 'seed': 42+i} for i in range(4)]
    module.atomic_json(tmp_path / 'run.json', {'planned': jobs})
    summary = {'metrics_valid': True, 'completed_trajectories': 4, 'per_task': {'a': {}}}
    module.atomic_json(tmp_path / 'summary.json', summary)
    assert module.check_summary(tmp_path, ['a']) == summary
    with pytest.raises(ValueError, match='plan differs'):
        module.check_summary(tmp_path, ['b'])
    summary['metrics_valid'] = False;module.atomic_json(tmp_path / 'summary.json', summary)
    with pytest.raises(ValueError, match='unscored'):
        module.check_summary(tmp_path, ['a'])


def test_occupied_gpu_never_launches_services(module, tmp_path, monkeypatch):
    c = module.Controller(tmp_path, tmp_path)
    monkeypatch.setattr(c, 'gpu_pids', lambda: {123})
    monkeypatch.setattr(c, 'launch', lambda *a, **k: pytest.fail('service launched'))
    with pytest.raises(RuntimeError, match='occupied'):
        c.evaluate()


def test_priority_and_frozen_command(module, tmp_path):
    assert module.SLOTS[:3] == (('sft',), ('e0',), ('e3',))
    assert sorted(a for slot in module.SLOTS for a in slot) == ['e0', 'e1', 'e3', 'sft']
    c = module.Controller(tmp_path, tmp_path)
    assert c.next_pending() == 'e2'
    assert c.next_pending() is None
    c.models['e3'] = tmp_path / 'e3'
    c.env['TAU3_USER_SERVED_MODEL_NAME'] = 'simulator'
    cmd = c.evaluation_command('e3', 8202, tmp_path / 'out', tmp_path / 'manifests', 4)
    assert cmd[cmd.index('--trials')+1] == '4'
    assert cmd[cmd.index('--target')+1] == 'selection'
    assert cmd[cmd.index('--policy-base-url')+1] == 'http://127.0.0.1:8202/v1'
    assert 'tau3-final' not in cmd


def test_hf_directory_with_only_config_is_not_a_merged_model(module, tmp_path):
    module.atomic_json(tmp_path / 'config.json', {'model_type': 'qwen3_5'})
    with pytest.raises(ValueError, match='lacks weights'):
        module.validate_hf(tmp_path)


def test_stop_only_targets_owned_process_group(module, tmp_path, monkeypatch):
    c = module.Controller(tmp_path, tmp_path)
    calls = []
    monkeypatch.setattr(module, 'stop_process', lambda proc, **kw: calls.append(proc))
    proc = object()
    c.children.append(proc)
    c.cleanup()
    assert calls == [proc]


def test_low_space_keeps_checkpoint_and_does_not_publish_backup(module, tmp_path, monkeypatch):
    run = checkpoint(module, tmp_path)
    out = tmp_path / 'post'
    out.mkdir()
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match='Insufficient persistent space'):
        module.persist_complete(tmp_path, out, 'e2')
    assert (run / 'global_step_20/data.pt').exists()
    assert not (tmp_path / 'persistent-checkpoints/e2_seed42').exists()


def reuse_fixture(module, tmp_path):
    plan = {'task_ids': ['a'], 'policy_temperature': .4, 'slots': [['sft']], 'first_free_slot_queue': []}
    module.atomic_json(tmp_path / 'plan.json', plan)
    module.atomic_json(tmp_path / 'software.json', {'evaluation_source_sha256': module.evaluation_source_hashes()})
    module.atomic_json(tmp_path / 'state.json', {'status': 'incomplete', 'models': {
        'sft': {'status': 'complete'}, **{a: {'status': 'failed'} for a in module.ARMS}}})
    (tmp_path / 'controller.exit').write_text('1')
    module.atomic_json(tmp_path / 'selection/sft/run.json', {'planned': [
        {'task_id': 'a', 'trial': i, 'seed': 42+i} for i in range(4)]})
    module.atomic_json(tmp_path / 'selection/sft/summary.json', {
        'metrics_valid': True, 'completed_trajectories': 4, 'per_task': {'a': {}},
        'metrics': {'pass@1': .5}})
    return dict(plan, slots=[['e0'], ['e1'], ['e2'], ['e3']], reuse_sft_from=str(tmp_path))


def test_recovery_reuses_sft_without_scheduling_more_sft_trials(module, tmp_path):
    plan = reuse_fixture(module, tmp_path)
    summary = module.validate_sft_reuse(tmp_path, plan)
    assert summary['metrics']['pass@1'] == .5
    c = module.Controller(tmp_path, tmp_path / 'new', tmp_path)
    assert c.slots == (('e0',), ('e1',), ('e2',), ('e3',))
    assert c.next_pending() is None


@pytest.mark.parametrize('change', ['protocol', 'source', 'attempted_trials', 'incomplete_sft'])
def test_recovery_refuses_incomparable_or_previously_scored_replays(module, tmp_path, change):
    plan = reuse_fixture(module, tmp_path)
    if change == 'protocol':
        plan['policy_temperature'] = .8
    elif change == 'source':
        module.atomic_json(tmp_path / 'software.json', {'evaluation_source_sha256': {}})
    elif change == 'attempted_trials':
        (tmp_path / 'smoke/e0').mkdir(parents=True)
    else:
        s = module.read(tmp_path / 'selection/sft/summary.json')
        s['metrics_valid'] = False
        module.atomic_json(tmp_path / 'selection/sft/summary.json', s)
    with pytest.raises(ValueError):
        module.validate_sft_reuse(tmp_path, plan)

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def continuation(tmp_path, monkeypatch):
    monkeypatch.setenv('TAU3_ROOT', str(tmp_path))
    monkeypatch.setenv('TAU3_RUN_ROOT', str(tmp_path / 'runs'))
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'env_info/a800_20260912'))
    module = importlib.import_module('run_e1_after_e0')
    monkeypatch.setattr(module.controller, 'W', tmp_path / 'runs')
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=200 * 2**30))
    return module


def checkpoint(root, step):
    actor = root / f'global_step_{step}' / 'actor'
    actor.mkdir(parents=True)
    for rank in range(4):
        for prefix in ('model', 'optim', 'extra_state'):
            (actor / f'{prefix}_world_size_4_rank_{rank}.pt').write_bytes(f'{prefix}-{step}-{rank}'.encode())
    (actor.parent / 'data.pt').write_bytes(b'dataloader')


def test_tmpfs_style_run_symlink_preserves_complete_boundary_rotation(tmp_path):
    from tau3_grpo.integrations.boundary_checkpoint import complete_boundary
    real = tmp_path / 'temporary' / 'e2_seed42'
    real.mkdir(parents=True)
    link = tmp_path / 'e2_seed42'
    link.symlink_to(real, target_is_directory=True)
    checkpoint(link, 10)
    complete_boundary(link, 10, 4)
    checkpoint(link, 20)
    complete_boundary(link, 20, 4)
    assert link.is_symlink()
    assert not (real / 'global_step_10').exists()
    assert (real / 'global_step_20/checkpoint-complete.json').is_file()
    assert (link / 'latest_checkpointed_iteration.txt').read_text().strip() == '20'


def test_durable_archive_has_weights_and_evidence_without_claiming_full_resume(continuation):
    source = continuation.controller.W / 'e2_seed42'
    checkpoint(source, 20)
    (source / 'metrics.jsonl').write_text('{"step":20}\n')
    (source / 'global_step_20/checkpoint-complete.json').write_text('{}')
    dest = Path(continuation.persist_tmpfs_models('e2', 20))
    receipt = json.loads((dest / 'archive.json').read_text())
    assert receipt['includes_optimizer'] is False
    assert len(receipt['model_sha256']) == 4
    assert not list(dest.rglob('optim_world_size*'))
    assert not (dest / 'global_step_20/checkpoint-complete.json').exists()
    assert len(list(source.rglob('optim_world_size*'))) == 4
    assert (dest / 'metrics.jsonl').read_text() == '{"step":20}\n'


def test_archive_corruption_is_detected_before_publishing(continuation, monkeypatch):
    source = continuation.controller.W / 'e3_seed42'
    checkpoint(source, 20)
    copy = continuation.shutil.copy2
    def corrupted(src, dst, *args, **kwargs):
        result = copy(src, dst, *args, **kwargs)
        if Path(src).name.startswith('model_world_size'):
            Path(dst).write_bytes(b'corrupted')
        return result
    monkeypatch.setattr(continuation.shutil, 'copy2', corrupted)
    with pytest.raises(AssertionError):
        continuation.persist_tmpfs_models('e3', 20)
    assert not (continuation.controller.W / 'persistent-models/e3_seed42').exists()
    assert len(list(source.rglob('model_world_size*'))) == 4

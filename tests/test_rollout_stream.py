import asyncio
import json
from types import SimpleNamespace

import pytest

from tau3_grpo.tracking.rollout_stream import save_completed_rollout


def output(value=1):
    return SimpleNamespace(model_dump=lambda: dict(response_ids=[value], response_mask=[1],
        extra_fields={'tool_receipts': [{'id': 'a'}, {'id': 'b'}]}, reward_score=0.5))


def test_completions_survive_pending_batch_and_duplicate_worker_indices(tmp_path, monkeypatch):
    monkeypatch.setenv('TAU3_ROLLOUT_STREAM_DIR', str(tmp_path))
    async def run():
        blocked = asyncio.Event()
        async def pending():
            await blocked.wait()
        task = asyncio.create_task(pending())
        paths = await asyncio.gather(*(save_completed_rollout(
            dict(step=0, sample_index=0, rollout_n=0, validate=True), output(i)) for i in range(12)))
        assert not task.done()
        assert len(set(paths)) == 12
        records = [json.loads(p.read_text()) for p in paths]
        assert sorted(r['output']['response_ids'][0] for r in records) == list(range(12))
        assert all(len(r['output']['extra_fields']['tool_receipts']) == 2 for r in records)
        assert all(not r['batch_complete'] and not r['resumable_checkpoint'] for r in records)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())
    assert not list(tmp_path.rglob('*.tmp'))


def test_serialization_failure_does_not_publish_partial_result(tmp_path, monkeypatch):
    monkeypatch.setenv('TAU3_ROLLOUT_STREAM_DIR', str(tmp_path))
    with pytest.raises(ValueError):
        asyncio.run(save_completed_rollout(dict(step=1, validate=False), output(float('nan'))))
    assert not list(tmp_path.rglob('*.json'))
    assert not list(tmp_path.rglob('*.tmp'))


def test_disabled_is_noop(monkeypatch):
    monkeypatch.delenv('TAU3_ROLLOUT_STREAM_DIR', raising=False)
    assert asyncio.run(save_completed_rollout({}, None)) is None

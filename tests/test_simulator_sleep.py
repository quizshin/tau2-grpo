import asyncio

import pytest

from tau3_grpo.training.rl.simulator_sleep import SimulatorSleep


class Session:
    def __init__(self, *, refuse=False):
        self.sleeping = False
        self.posts = []
        self.refuse = refuse

    def get(self, url, **kwargs):
        return self

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        if not self.refuse:
            self.sleeping = url.endswith('/sleep')
        return self

    def raise_for_status(self):
        pass

    def json(self):
        return {'is_sleeping': self.sleeping}


def test_confirmed_sleep_and_wake_are_idempotent(tmp_path):
    session = Session()
    manager = SimulatorSleep('http://127.0.0.1:8100/v1', tmp_path / 'events.jsonl', session=session)
    manager.set_sleeping(True)
    manager.set_sleeping(True)
    manager.set_sleeping(False)
    assert len(session.posts) == 2
    assert session.posts[0][1]['params'] == {'level': 1, 'mode': 'wait'}
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 3


def test_unconfirmed_memory_release_stops_training(tmp_path):
    manager = SimulatorSleep('http://127.0.0.1:8100/v1', tmp_path / 'events.jsonl', session=Session(refuse=True))
    with pytest.raises(RuntimeError, match='requested memory state'):
        manager.set_sleeping(True)
    assert not (tmp_path / 'events.jsonl').exists()
    with pytest.raises(ValueError, match='owned local'):
        SimulatorSleep('http://other-host:8100/v1', tmp_path / 'events.jsonl')


@pytest.mark.parametrize('failed', [False, True])
def test_native_rollout_releases_simulator_on_success_and_failure(monkeypatch, failed):
    from tau3_grpo.integrations.verl.sleeping_simulator import (
        AgentLoopManager,
        SleepingSimulatorAgentLoopManager,
    )
    calls = []
    async def native(self, prompts):
        calls.append('generate')
        if failed:
            raise RuntimeError('rollout failed')
        return prompts
    monkeypatch.setattr(AgentLoopManager, 'generate_sequences', native)
    manager = object.__new__(SleepingSimulatorAgentLoopManager)
    class Memory:
        def set_sleeping(self, value):
            calls.append(value)
    manager.simulator_memory = Memory()
    async def check():
        if failed:
            with pytest.raises(RuntimeError, match='rollout failed'):
                await manager.generate_sequences('batch')
        else:
            assert await manager.generate_sequences('batch') == 'batch'
    asyncio.run(check())
    assert calls == [False, 'generate', True]

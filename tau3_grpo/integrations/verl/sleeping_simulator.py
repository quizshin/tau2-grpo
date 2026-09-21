"""Reuse the native agent loop, releasing simulator memory between rollouts."""
import asyncio
import os
from pathlib import Path

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, auto_await

from tau3_grpo.training.rl.simulator_sleep import SimulatorSleep


class SleepingSimulatorAgentLoopManager(AgentLoopManager):
    @classmethod
    @auto_await
    async def create(cls, config, worker_group=None, rollout_resource_pool=None,
                     reward_loop_worker_handles=None):
        if os.getenv('TAU3_SIMULATOR_COLOCATED_SLEEP') != '1':
            raise ValueError('Simulator phase sharing must be explicitly enabled')
        memory = SimulatorSleep(os.environ['TAU3_USER_BASE_URL'],
            Path(config.trainer.default_local_dir) / 'simulator-memory.jsonl')
        # Free simulator weights before initializing policy rollout replicas.
        await asyncio.to_thread(memory.set_sleeping, True)
        instance = await super().create(config, worker_group, rollout_resource_pool,
                                        reward_loop_worker_handles)
        instance.simulator_memory = memory
        return instance

    @auto_await
    async def generate_sequences(self, prompts):
        await asyncio.to_thread(self.simulator_memory.set_sleeping, False)
        try:
            return await super().generate_sequences(prompts)
        finally:
            # Applies to both training and validation, including failed calls.
            await asyncio.to_thread(self.simulator_memory.set_sleeping, True)

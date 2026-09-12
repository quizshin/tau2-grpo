"""Registry/tool isolation under async interleaving; environment/model are stubs."""
import asyncio
from types import SimpleNamespace

import pytest

from test_integration_layer import factory, interaction, _tool, _AgentData
from tau3_grpo.envs.registry import SESSIONS


@pytest.mark.parametrize('concurrency',[4,16,32])
def test_many_rollouts_interleaved_and_cancelled(interaction,concurrency):
    async def run():
        ready=asyncio.Event()
        active=0
        seen={}
        async def trajectory(index):
            nonlocal active
            rid=f'stress-{index}'
            await interaction.start_interaction(rid,task_id='airline_1')
            active+=1
            if active==concurrency:
                ready.set()
            tool=_tool('book_reservation')
            instance,_=await tool.create()
            try:
                await ready.wait()
                for turn in range(3):
                    calls=interaction.record_tool_batch(rid,tool_calls=[SimpleNamespace(
                        name='book_reservation',arguments='{}') for _ in range(index%4+1)])
                    for call in calls:
                        await asyncio.sleep(0)
                        await tool.execute(instance,{},agent_data=_AgentData(rid),recorded_tool_call=call)
                s=SESSIONS.require(rid).session
                seen[rid]=(id(s.db),s.db.payload['n'],s.tool_calls,s.assistant_turns)
                assert s.db.payload['n']==3*(index%4+1)
                assert s.tool_calls==3*(index%4+1) and s.assistant_turns==3
            finally:
                await tool.release(instance)
                await interaction.finalize_interaction(rid)
        tasks=[asyncio.create_task(trajectory(i)) for i in range(concurrency)]
        await ready.wait()
        tasks[0].cancel()
        results=await asyncio.gather(*tasks,return_exceptions=True)
        assert isinstance(results[0],asyncio.CancelledError)
        assert all(x is None for x in results[1:])
        assert len(seen)==concurrency-1
        assert len({x[0] for x in seen.values()})==len(seen)
        assert SESSIONS.active_count()==0
    asyncio.run(run())

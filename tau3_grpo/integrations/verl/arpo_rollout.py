"""ARPO group scheduling; actual transitions stay in the native ToolAgentLoop."""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4

from tau3_grpo.algorithms.arpo import branch_probability, group_rows, stable_seed
from tau3_grpo.envs.registry import SESSIONS, SessionEntry
from tau3_grpo.envs.session import TrajectorySession


def capture(data):
    """Called only between completed transitions, never during a tool request."""
    entry = SESSIONS.require(data.request_id)
    state = deepcopy({k: v for k, v in vars(data).items() if k != 'interaction'})
    entry_state = deepcopy({k: v for k, v in vars(entry).items() if k != 'session'})
    return type(data), state, data.interaction, entry.session.snapshot(), entry_state


def restore(snapshot, request_id):
    cls, state, interaction, session, entry_state = snapshot
    child = cls.__new__(cls)
    child.__dict__.update(deepcopy(state))
    child.interaction = interaction
    child.request_id = request_id
    child.metrics = {}  # Shared prefix cost was already paid by its source.
    child.arpo_after_tools = False  # Do not recursively branch the same boundary.
    SESSIONS.register(request_id, SessionEntry(
        session=TrajectorySession.from_snapshot(session, session_id=request_id), **deepcopy(entry_state)))
    return child


class GroupRollout:
    def __init__(self, config, size, uid, seed, policy_step):
        if not 1 <= config.initial_rollouts <= size or size < 2:
            raise ValueError('ARPO initial roots must fit a group of at least two')
        self.config, self.size, self.uid = config, size, str(uid)
        self.seed, self.policy_step = int(seed), int(policy_step)
        self.rng = random.Random(stable_seed(seed, policy_step, 'branch'))
        self.pending = deque((i, None, None) for i in range(config.initial_rollouts))
        self.allocated = config.initial_rollouts
        self.current = None
        self.active_ids = set()
        self.decisions = []
        self.nodes = {}

    def attach(self, data):
        index, parent = self.current
        self.active_ids.add(data.request_id)
        inherited = getattr(data, 'arpo_node', None)
        prefix = list(data.prompt_ids)
        data.arpo_node = dict(node=index, parent=parent,
            root=inherited['root'] if inherited else index,
            shared_response_tokens=len(data.response_mask) if inherited else 0,
            prefix_sha256=hashlib.sha256(json.dumps(prefix).encode()).hexdigest() if inherited else None,
            prefix_tool_calls=SESSIONS.require(data.request_id).session.tool_calls,
            prefix_user_turns=SESSIONS.require(data.request_id).session.user_turns,
            new_tokens=0, entropy_measurements=[])
        self.nodes[index] = data.arpo_node
        data.arpo_after_tools = False
        if not inherited:
            data.arpo_initial_entropy = None
        # Each continuation gets an independent simulator request seed; local
        # simulator state/RNG was inherited before this explicit branch split.
        SESSIONS.require(data.request_id).session.set_seed(stable_seed(self.seed, self.policy_step, index, 'user'))

    def sampling_params(self, data, params):
        return dict(params, arpo_entropy_top_k=self.config.entropy_top_k,
                    arpo_entropy_window=self.config.entropy_window,
                    seed=stable_seed(self.seed, self.policy_step,
                                     data.arpo_node['node'], data.assistant_turns, 'actor'))

    def before_generation(self, data):
        data.arpo_last_entropy = None
        if data.arpo_after_tools and self.allocated < self.size:
            return capture(data)
        return None

    def after_generation(self, data, snapshot):
        measurement = data.arpo_last_entropy
        if measurement is None:
            # A native turn/context limit can terminate without a generation.
            if data.termination_reason:
                return
            raise ValueError('ARPO rollout did not return entropy measurements')
        current = float(measurement['value'])
        data.arpo_node['entropy_measurements'].append(deepcopy(measurement))
        data.arpo_node['new_tokens'] += int(measurement['generated_tokens'])
        if data.arpo_initial_entropy is None:
            data.arpo_initial_entropy = current
        if snapshot is None:
            return
        probability = branch_probability(current, data.arpo_initial_entropy, self.config)
        draw = self.rng.random()
        take = draw < probability
        decision = dict(node=data.arpo_node['node'], turn=data.assistant_turns,
                        initial=data.arpo_initial_entropy, current=current,
                        probability=probability, draw=draw, branched=take,
                        prefix_tokens=len(snapshot[1]['response_mask']))
        self.decisions.append(decision)
        if take:
            slot = self.allocated
            self.allocated += 1
            child = restore(snapshot, uuid4().hex)
            self.active_ids.add(child.request_id)
            self.pending.append((slot, child, data.arpo_node['node']))
            decision['child'] = slot

    def finish(self, data):
        session = SESSIONS.require(data.request_id).session
        data.arpo_node['new_tool_calls'] = session.tool_calls - data.arpo_node['prefix_tool_calls']
        data.arpo_node['new_user_turns'] = session.user_turns - data.arpo_node['prefix_user_turns']
        data.arpo_node['simulator_seed'] = session.seed

    def publish(self, data, output):
        info = dict(schema='arpo_tau_rollout_v1', settings=asdict(self.config),
                    config_sha256=self.config.identity, group_uid=self.uid,
                    group_size=self.size, policy_step=self.policy_step, seed=self.seed,
                    **data.arpo_node)
        output.extra_fields['arpo_rollout_json'] = json.dumps(info, sort_keys=True, allow_nan=False)
        output.extra_fields.setdefault('reward_extra_info', {})['arpo_rollout_json'] = output.extra_fields['arpo_rollout_json']

    async def run(self, run_one):
        outputs = [None] * self.size
        try:
            while any(output is None for output in outputs):
                if not self.pending:
                    slot = self.allocated
                    self.allocated += 1
                    if slot >= self.size:
                        raise RuntimeError('ARPO lost a scheduled trajectory')
                    self.pending.append((slot, None, None))
                slot, resume, parent = self.pending.popleft()
                self.current = (slot, parent)
                outputs[slot] = await run_one(slot, resume, self)
            for slot, out in enumerate(outputs):
                node = self.nodes[slot]
                if node['parent'] is None:
                    continue
                parent = outputs[node['parent']]
                n = node['shared_response_tokens']
                if (out.prompt_ids != parent.prompt_ids or out.response_ids[:n] != parent.response_ids[:n]
                        or out.response_mask[:n] != parent.response_mask[:n]
                        or out.response_logprobs is None or parent.response_logprobs is None
                        or out.response_logprobs[:n] != parent.response_logprobs[:n]):
                    raise ValueError('ARPO shared prefix tokens/mask/logprobs changed')
            # Identical group audit on all rows survives row redistribution.
            raw = json.dumps(dict(decisions=self.decisions, group_size=self.size,
                                  nodes=self.nodes), sort_keys=True, allow_nan=False)
            for out in outputs:
                out.extra_fields['arpo_group_json'] = raw
                out.extra_fields.setdefault('reward_extra_info', {})['arpo_group_json'] = raw
            return outputs
        finally:
            for request_id in self.active_ids:
                SESSIONS.pop(request_id)


async def generate_groups(worker, batch, sampling_params, trajectory_info, traced_indices):
    """Preserve input row order; each entire UID group lives on this worker."""
    from tau3_grpo.integrations.verl.arpo import validate_training_config

    config = validate_training_config(worker.config)
    size = int(worker.rollout_config.n)
    rows_by_group = group_rows(batch.non_tensor_batch['uid'], size)
    outputs = [None] * len(batch)

    async def one_group(rows):
        first = rows[0]
        uid = str(batch.non_tensor_batch['uid'][first])
        # Never combine different tasks/prompts under a malformed UID.
        for name in ('raw_prompt', 'extra_info', 'agent_name'):
            values = batch.non_tensor_batch.get(name)
            if values is not None and any(repr(values[i]) != repr(values[first]) for i in rows):
                raise ValueError(f'ARPO group contains inconsistent {name}')
        group_seed = stable_seed(worker.config.data.seed, batch.meta_info.get('global_steps', -1),
                                 int(batch.non_tensor_batch['arpo_group_index'][first]))
        controller = GroupRollout(config, size, uid, group_seed,
                                  batch.meta_info.get('global_steps', -1))

        async def run_one(slot, resume, control):
            i = rows[slot]
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            if kwargs.get('agent_name') != 'tool_agent':
                raise ValueError('ARPO τ requires tool_agent')
            # Retain native output until group audit is complete; postprocess below.
            return await worker._run_agent_loop(
                sampling_params, trajectory_info[i], trace=i in traced_indices,
                _tau3_arpo=control, _tau3_resume=resume, _tau3_raw_output=True, **kwargs)

        raw_outputs = await controller.run(run_one)
        from tau3_grpo.tracking.rollout_stream import save_completed_rollout
        for i, raw in zip(rows, raw_outputs, strict=True):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            outputs[i] = await worker._agent_loop_postprocess(raw, **kwargs)
            await save_completed_rollout(trajectory_info[i], raw)

    tasks = [asyncio.create_task(one_group(rows)) for rows in rows_by_group]
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return outputs

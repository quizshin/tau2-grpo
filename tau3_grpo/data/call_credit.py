"""Join paid rewards to exact generated call spans by runtime identity."""
from __future__ import annotations

import json

import numpy as np

from tau3_grpo.data.call_attribution import retained_call_attribution


def call_credit_inputs(processes, raw_facts, uids, response_ids, response_mask):
    ids, mask = np.asarray(response_ids), np.asarray(response_mask)
    if ids.shape != mask.shape or ids.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError('Call credit requires actual response IDs and binary mask')
    if any(len(x) != len(mask) for x in (processes, raw_facts, uids)):
        raise ValueError('Call credit metadata batch length mismatch')
    rewards, spans, eligible, receipts = [], [], [], []
    seen_trajectories = set()
    for i, process in enumerate(processes):
        rewards.append([])
        spans.append([])
        eligible.append([])
        receipts.append([])
        if not mask[i].any():
            continue
        facts = json.loads(raw_facts[i])
        if facts.get('schema') != 'tau3_trajectory_facts_v1':
            raise ValueError('Missing call-credit trajectory facts schema')
        identity = facts.get('identity', {})
        trajectory = identity.get('trajectory_id')
        if (not trajectory or trajectory in seen_trajectories
                or trajectory != process.get('trajectory_id')
                or str(identity.get('sample_group_uid')) != str(uids[i])):
            raise ValueError('Call-credit trajectory/UID identity mismatch')
        seen_trajectories.add(trajectory)
        tokens = facts['tokens']
        actual_ids, actual_mask = tokens['response_ids'], tokens['response_mask']
        length = len(actual_ids)
        if (length > ids.shape[1] or len(actual_mask) != length
                or not np.array_equal(actual_ids, ids[i, :length])
                or not np.array_equal(actual_mask, mask[i, :length]) or mask[i, length:].any()):
            raise ValueError('Call-credit actual token identity/mask mismatch')
        turns = process['turn_records']
        if len(turns) != len(facts['turns']):
            raise ValueError('Call-credit turn count mismatch')
        if (process['settings']['mode'] == 'paper'
                and process['settings']['paper_options']['aggregation'] != 'sum'):
            raise ValueError('Call credit requires sum aggregation')
        row_ids = set()
        for k, (turn, fact) in enumerate(zip(turns, facts['turns'], strict=True)):
            if (turn['turn_index'] != k or fact['turn_index'] != k
                    or turn['token_span'] != fact['token_span']
                    or turn['token_span'] != process['turn_spans'][k]
                    or turn['reward'] != process['turn_rewards'][k]):
                raise ValueError('Call-credit turn identity/reward mismatch')
            attr = fact.get('call_attribution')
            if not attr or attr.get('schema') != 'tau3_call_attribution_v1':
                raise ValueError('Call credit requires TAU3_RECORD_CALL_ATTRIBUTION=1 on workers')
            if attr.get('coordinate_system') != 'response_token_offset_half_open':
                raise ValueError('Unknown call-credit coordinate system')
            if attr['emitted_span'] != fact['token_span']:
                raise ValueError('Call-credit attribution turn mismatch')
            checked = retained_call_attribution(attr, actual_ids, actual_mask)
            if (checked['retained_span'] != attr['retained_span']
                    or checked['eligible_for_call_credit'] != attr['eligible_for_call_credit']):
                raise ValueError('Call-credit attribution eligibility mismatch')
            by_id = {}
            for event in fact['tool_calls']:
                key = event.get('id')
                if not key or key in row_ids:
                    raise ValueError('Missing or duplicate executed call ID')
                row_ids.add(key)
                by_id[key] = event
            paid = {}
            for event in turn['tool_calls']:
                key = event.get('id')
                if key in paid or key not in by_id:
                    raise ValueError('Paid call ID differs from execution facts')
                if any(event.get(f) != by_id[key].get(f)
                       for f in ('name', 'arguments', 'error', 'db_hash_after')):
                    raise ValueError('Paid call content differs from execution facts')
                paid[key] = event
            if set(paid) != set(by_id):
                raise ValueError('Paid calls do not cover executed call IDs')
            attr_ids = [c['call_id'] for c in checked['calls'] if c['call_id'] is not None]
            executed = [c for c in checked['calls'] if c['execution_status'] == 'executed']
            if (len(attr_ids) != len(set(attr_ids))
                    or {c['call_id'] for c in executed} != set(paid)
                    or len(executed) != len(paid)):
                raise ValueError('Attribution call IDs do not match executed paid calls')
            for call in executed:
                event = paid[call['call_id']]
                if any(call.get(f) != event.get(f) for f in ('name', 'error', 'db_hash_after')):
                    raise ValueError('Attribution call receipt mismatch')
            # Reward order follows native execution, NOT reward tier or tool name.
            q = [float(paid[c['call_id']]['reward']) for c in executed]
            ok = checked['eligible_for_call_credit']
            exact_spans = None
            if ok:
                if checked['scan_status'] != 'exact' or checked['parse_status'] != 'parsed':
                    raise ValueError('Eligible attribution lacks exact parser provenance')
                exact_spans = [checked['blocks'][c['block_index']]['retained_span'] for c in executed]
            rewards[i].append(q)
            spans[i].append(exact_spans)
            eligible[i].append(ok)
            receipts[i].append({'call_ids': [c['call_id'] for c in executed],
                                'eligible': ok, 'scan_status': checked['scan_status'],
                                'alignments': [c['alignment'] for c in checked['calls']]})
    return dict(call_rewards=rewards, call_spans=spans, eligible_turns=eligible), receipts

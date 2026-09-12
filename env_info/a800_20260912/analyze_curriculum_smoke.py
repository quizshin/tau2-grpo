"""Read-only analysis of the one-step online run; never launches training."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def analyze(run_dir):
    import torch
    from verl import DataProto

    result = json.loads((run_dir / 'online-result.json').read_text())
    manifest = json.loads((run_dir / 'online-smoke/experiment_manifest.json').read_text())
    rows = [json.loads(line) for line in
            (run_dir / 'online-smoke/rollouts/1.jsonl').read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        groups[row['task_id']].append(row)
    expected = manifest['schedule'][0]['task_ids']
    assert set(groups) == set(expected) and len(rows) == 64
    assert all(len(group) == 8 for group in groups.values())
    assert all(row['score'] in (0, 1) for row in rows)
    # Turn/context limits receive zero without reaching the verifier.
    assert all(row['scored'] or (row['score'] == 0 and row['termination_reason'] in
                               ('context_window_exceeded', 'max_steps')) for row in rows)
    assert {row['tool_protocol'] for row in rows} == {'airline_sequential_multicall_v1'}
    assert len({json.loads(row['trajectory_json'])['session_id'] for row in rows}) == 64

    packet = run_dir / 'online-smoke/update-batches/update_000001.pkl'
    data = DataProto.load_from_disk(str(packet))
    batch = data.batch
    mask = batch['response_mask'].bool()
    assert len(batch) == 64 and not any(data.non_tensor_batch['tau3_is_padding'])
    for name in ('old_log_probs', 'ref_log_prob', 'advantages', 'returns'):
        assert torch.isfinite(batch[name][mask]).all(), name
    scores = batch['token_level_scores'].sum(-1)
    group_reports = []
    for task_id in expected:
        indices = [i for i, value in enumerate(data.non_tensor_batch['task_id']) if value == task_id]
        assert len(indices) == 8
        rewards = sorted(row['score'] for row in groups[task_id])
        assert sorted(scores[indices].tolist()) == rewards
        uid_values = {str(data.non_tensor_batch['uid'][i]) for i in indices}
        assert len(uid_values) == 1
        active = int(((batch['advantages'][indices] != 0) & mask[indices]).sum())
        label = 'all_zero' if sum(rewards) == 0 else 'all_one' if sum(rewards) == 8 else 'mixed'
        if label != 'mixed':
            assert active == 0
        group_reports.append({'task_id': task_id, 'successes': int(sum(rewards)),
                              'samples': 8, 'kind': label, 'active_advantage_tokens': active,
                              'failure_categories': dict(Counter(row['failure_category'] for row in groups[task_id]))})

    log = re.sub(r'\x1b\[[0-9;]*m', '', (run_dir / 'online.log').read_text())
    metric_lines = [line for line in log.splitlines() if 'timing_s/step:' in line]
    assert len(metric_lines) == 1
    line = metric_lines[0]

    def metric(key):
        pattern = re.escape(key) + r':\s*(?:np\.(?:float64|float32|int64|int32)\()?([-+0-9.eE]+)'
        return float(re.search(pattern, line).group(1))

    timings = {key: metric('timing_s/' + key) for key in
               ('gen', 'old_log_prob', 'ref', 'update_actor', 'update_weights', 'step')}
    assert metric('actor/grad_norm') > 0
    worker_audits = defaultdict(list)
    for path in (run_dir / 'online-smoke/weight-audits').glob('*.json'):
        worker_audits[path.name.split('-')[2]].append(json.loads(path.read_text()))
    assert len(worker_audits) == 4
    weight_reports = []
    for pid, audits in sorted(worker_audits.items()):
        audits.sort(key=lambda audit: audit['time'])
        assert len(audits) == 2 and all(audit['all_conv_match'] for audit in audits)
        before, after = [{entry['key']: entry['sha256'] for entry in audit['conv']} for audit in audits]
        changed = sum(before[key] != after[key] for key in before)
        assert changed > 0
        weight_reports.append({'pid': pid, 'exact_incoming_weight_matches': True,
                               'changed_conv_tensors': changed, 'audited_conv_tensors': len(before)})

    budgets = []
    for steps in (10, 15, 20, 50):
        hours = steps * timings['step'] / 3600
        budgets.append({'steps': steps, 'trajectories': steps * 64,
                        'trajectories_per_task': steps * 64 / 40,
                        'training_hours_single_step_extrapolation': hours,
                        'training_hours_if_steps_20_percent_slower': hours * 1.2})
    summary = {**result, 'timings_seconds': timings,
               'startup_and_other_process_seconds': result['train_process_wall_seconds'] - timings['step'],
               'model': 'new-off SFT, dense Qwen3.5-4B full language parameters, thinking disabled',
               'grad_norm': metric('actor/grad_norm'),
               'actor_reported_memory_gib': {'allocated': metric('perf/max_memory_allocated_gb'),
                                            'reserved': metric('perf/max_memory_reserved_gb')},
               'memory_scope': 'Trainer-reduced actor allocator peaks; not total nvidia-smi use or a per-rank maximum guarantee.',
               'scorer_successes': int(sum(row['score'] for row in rows)),
               'scorer_success_rate': sum(row['score'] for row in rows) / len(rows),
               'verifier_scored_trajectories': sum(bool(row['scored']) for row in rows),
               'zero_reward_without_verifier': sum(not row['scored'] for row in rows),
               'group_kinds': dict(Counter(row['kind'] for row in group_reports)),
               'groups': group_reports,
               'policy_tokens': int(mask.sum()),
               'active_advantage_tokens': int(((batch['advantages'] != 0) & mask).sum()),
               'failure_categories': dict(Counter(row['failure_category'] for row in rows)),
               'termination_reasons': dict(Counter(row['termination_reason'] for row in rows)),
               'tool_parser_set_serialization_error_observed': 'Object of type set is not JSON serializable' in log,
               'unique_session_ids': 64, 'worker_weight_sync_audits': weight_reports,
               'budget_estimates': budgets,
               'limitations': ['One initial-SFT batch, not a speed A/B or independent evaluation.',
                               'No next rollout after the updated weights and no resume/checkpoint exercised.',
                               'Scorer success is not independent semantic/policy correctness.',
                               'Budget estimates exclude startup, simulator startup, checkpointing and evaluation.'],
               'artifact_sha256': {str(path.relative_to(run_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (packet, run_dir / 'online.log', run_dir / 'online-smoke/rollouts/1.jsonl')}}
    (run_dir / 'online-analysis.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    analyze(parser.parse_args().run_dir)

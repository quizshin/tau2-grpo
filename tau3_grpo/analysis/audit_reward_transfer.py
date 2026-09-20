"""Apply one fixed reward recipe to historical buffers, without fitting.

Logged process records are replayed before rescoring. Standalone evaluation
messages are paired by tool ID in execution order; synthetic spans are explicitly
not token spans and are never exported as training replay. Missing evidence is
reported as an error, not interpreted as a successful tool call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

from tau3_grpo.analysis.audit_paper_credit import association
from tau3_grpo.analysis.calibrate_paper_rewards import implementation_hashes
from tau3_grpo.evaluation.process_reward import (
    READ_TOOLS,
    WRITE_TOOLS,
    reward_settings,
    score_turns,
)


def simulation_turns(simulation):
    """Preserve assistant rounds/multicalls and require complete tool evidence."""
    turns, pending, seen = [], {}, set()
    for message in simulation['messages']:
        role = message['role']
        if role == 'assistant':
            if pending:
                raise ValueError('assistant continued before all tool responses')
            k = len(turns)
            turn = {'schema': 'tau3_turn_v1', 'turn_index': k,
                    'token_span': [k, k + 1], 'tool_calls': []}
            for call in message.get('tool_calls') or []:
                ident = call.get('id')
                if not isinstance(ident, str) or not ident or ident in seen:
                    raise ValueError('missing or duplicate tool call ID')
                if call.get('requestor', 'assistant') != 'assistant':
                    raise ValueError('unsupported tool requestor')
                if not isinstance(call.get('name'), str) or not isinstance(call.get('arguments'), dict):
                    raise ValueError('tool name/arguments are not structured')
                event = {'name': call['name'], 'arguments': deepcopy(call['arguments']), 'id': ident}
                pending[ident] = event
                seen.add(ident)
                turn['tool_calls'].append(event)
            turns.append(turn)
        elif role == 'tool':
            if message.get('requestor', 'assistant') != 'assistant':
                raise ValueError('unsupported tool response requestor')
            ident = message.get('id')
            if ident not in pending:
                raise ValueError('orphan or duplicate tool response')
            if ident != next(iter(pending)):
                raise ValueError('tool response order differs from sequential call execution')
            if type(message.get('error')) is not bool or message.get('content') is None:
                raise ValueError('tool response lacks explicit error/content evidence')
            event = pending.pop(ident)
            event.update(error=message['error'], observation=message['content'],
                         observation_truncated=bool(message.get('observation_truncated', False)))
        elif role == 'user':
            if pending:
                raise ValueError('user message before all tool responses')
        else:
            raise ValueError(f'unsupported simulation message role: {role}')
    if pending or not turns:
        raise ValueError('incomplete simulation tool responses or no assistant turns')
    return turns


def gold_signature(actions):
    return [{'name': g['name'], 'arguments': g['arguments']}
            for g in actions if g.get('requestor', 'assistant') == 'assistant']


def adapt_row(row, entry, recipe):
    if row['task_id'] != entry['task_id']:
        raise ValueError('task identity mismatch')
    gold = entry['task']['evaluation_criteria'].get('actions') or []
    if row.get('process_reward_json'):
        p = json.loads(row['process_reward_json'])
        if gold_signature(p['golden_actions']) != gold_signature(gold):
            raise ValueError('saved gold differs from declared manifest')
        outcome = row['score']
        if type(row.get('scored')) is not bool:
            raise ValueError('missing scored status')
        scored = row['scored']
        if outcome not in (0, 1) or (not scored and outcome != 0):
            raise ValueError('invalid recorded outcome')
        if p['official_outcome'] != outcome:
            raise ValueError('saved process outcome differs from row')
        original = score_turns(p['turn_records'], p['golden_actions'], p['reward_basis'],
                               p['settings'], official_outcome=outcome)
        if original['turn_rewards'] != p['turn_rewards'] or original['turn_spans'] != p['turn_spans']:
            raise ValueError('original process rewards/spans do not replay')
        process = score_turns(p['turn_records'], gold, p['reward_basis'], recipe, official_outcome=outcome)
        kind = 'logged_process_replayed'
    elif 'simulation' in row:
        sim = row['simulation']
        if sim['task_id'] != row['task_id'] or sim.get('seed') != row.get('seed'):
            raise ValueError('simulation identity/seed mismatch')
        info = sim.get('reward_info')
        if not info or row.get('reward') not in (0, 1) or info.get('reward') != row['reward']:
            raise ValueError('missing or inconsistent official simulation reward')
        reason = sim.get('termination_reason')
        if not reason or reason != row.get('termination_reason'):
            raise ValueError('simulation termination reason mismatch')
        # Pinned official evaluator's two scorable stop reasons. Avoid importing
        # the live verifier/environment package in this offline reader.
        outcome, scored = row['reward'], reason in {'agent_stop', 'user_stop'}
        basis = info.get('reward_basis')
        # Official evaluator returns a zero with null basis on premature stops.
        # A stored trajectory is not necessarily a completed official DB score.
        if scored and not isinstance(basis, list):
            raise ValueError('completed simulation lacks official reward basis')
        if not scored:
            if outcome != 0 or basis is not None:
                raise ValueError('premature simulation has inconsistent reward evidence')
            basis = []
        process = score_turns(simulation_turns(sim), gold, basis, recipe,
                              official_outcome=outcome)
        kind = 'simulation_synthetic_turn_spans_no_training_replay'
    else:
        raise ValueError('legacy text-only row lacks structured tool evidence; cannot fabricate replay')
    return process, outcome, scored, kind


def feature_row(process, *, task_id, outcome, scored):
    turns = process['turn_records']
    if not turns:
        raise ValueError('empty process')
    parts, calls, tools = Counter(), Counter(), Counter()
    for turn in turns:
        events = turn['tool_calls']
        divisor = max(1, len(events)) if process['settings']['paper_options']['aggregation'] == 'mean' else 1
        if not events:
            parts['message'] += turn['reward']
        for event in events:
            tier, name = event['reward_type'], event['name']
            kind = ('read' if name in READ_TOOLS or name == 'get_flight_status'
                    else 'write' if name in WRITE_TOOLS else 'other')
            component = f'gold_{kind}' if tier == 'gold_exact' else tier
            if tier == 'gold_exact' and process['settings']['version'] == 'paper_env_split_v3':
                component = 'gold_other'  # split scorer keeps non-DB exact actions separate
            parts[component] += event['reward'] / divisor
            calls[component] += 1
            tools[f'{tier}:{name}'] += 1
    total = sum(process['turn_rewards'])
    if not np.isclose(sum(parts.values()), total, atol=1e-10, rtol=1e-12):
        raise ValueError('component rewards do not reconstruct total')
    result = {'task_id': task_id, 'outcome': float(outcome), 'scored': scored,
              'turn_count': len(turns), 'mean_reward': total / len(turns), 'total_reward': total,
              'calls': dict(calls), 'tools': dict(tools)}
    for part in ('gold_read', 'gold_write', 'gold_other', 'soft_match', 'error',
                 'state_change', 'duplicate', 'read_only', 'unknown', 'message'):
        result[part + '_mean'] = parts[part] / len(turns)
        result[part + '_calls'] = calls[part]
    return result


def describe(rows):
    # Scored-only is a sensitivity report, not a replacement outcome definition.
    features = ['mean_reward', 'total_reward', 'turn_count'] + [
        f'{part}_{suffix}' for part in ('gold_read', 'gold_write', 'error', 'state_change', 'soft_match')
        for suffix in ('mean', 'calls')]
    result = {}
    for name, subset in [('recorded_outcomes', rows), ('officially_scored_only', [r for r in rows if r['scored']])]:
        stats = {}
        for feature in features if subset else []:
            stat = association(subset, feature)
            # Reward components can be negative; >0 is not "present" here.
            stat.pop('present')
            stat.pop('absent')
            stat.update(positive=sum(r[feature] > 0 for r in subset),
                        zero=sum(r[feature] == 0 for r in subset),
                        negative=sum(r[feature] < 0 for r in subset))
            stats[feature] = stat
        result[name] = {'rows': len(subset), 'tasks': len({r['task_id'] for r in subset}),
                        'successes': sum(r['outcome'] for r in subset),
                        'features': stats}
    result['calls'] = dict(sum((Counter(r['calls']) for r in rows), Counter()))
    result['tools'] = dict(sum((Counter(r['tools']) for r in rows), Counter()))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe-report', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'selection'), required=True)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError('use a new output directory')
    raw = args.recipe_report.read_bytes()
    report = json.loads(raw)
    if report['implementation_sha256'] != implementation_hashes():
        raise ValueError('candidate implementation changed; re-audit provenance first')
    recipe = reward_settings(report['rounds'][-1]['candidate_recipe'])
    manifest_raw = args.manifest.read_bytes()
    entries = [json.loads(line) for line in manifest_raw.splitlines() if line.strip()]
    if not entries or any(e['split'] != args.split for e in entries):
        raise ValueError('manifest split mismatch; final data not accepted')
    manifest = {e['task_id']: e for e in entries}
    if len(manifest) != len(entries):
        raise ValueError('duplicate manifest task')
    result = {'schema': 'reward_transfer_diagnostic_v1', 'status': 'diagnostic_only', 'split': args.split,
              'recipe_report_sha256': hashlib.sha256(raw).hexdigest(), 'recipe': recipe,
              'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
              'implementation_sha256': implementation_hashes(),
              'audit_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'note': 'No weights fitted; no training replay or acceptance claim for adapted simulations.',
              'datasets': []}
    detail_rows = []
    seen = set()
    for path in args.input:
        buffer = path.read_bytes()
        sha = hashlib.sha256(buffer).hexdigest()
        if sha in seen:
            raise ValueError('duplicate input buffer')
        seen.add(sha)
        rows = [json.loads(line) for line in buffer.splitlines() if line.strip()]
        if any(r['task_id'] not in manifest for r in rows):
            raise ValueError('input task outside declared manifest')
        features, rejected, modes = [], [], Counter()
        for i, row in enumerate(rows):
            try:
                p, outcome, scored, mode = adapt_row(row, manifest[row['task_id']], recipe)
                feature = feature_row(p, task_id=row['task_id'], outcome=outcome, scored=scored)
                feature.update(source=str(path), row=i, termination_reason=row.get('termination_reason'))
                features.append(feature)
                modes[mode] += 1
            except (KeyError, ValueError, TypeError) as exc:
                rejected.append({'row': i, 'task_id': row.get('task_id'), 'reason': str(exc)})
        detail_rows.extend(features)
        counts = Counter(r['task_id'] for r in rows)
        result['datasets'].append({'path': str(path), 'sha256': sha, 'input_rows': len(rows),
                                   'task_sample_count_histogram': dict(Counter(counts.values())),
                                   'manifest_tasks_without_rows': sorted(set(manifest) - set(counts)),
                                   'adapted_rows': len(features), 'modes': dict(modes),
                                   'rejected': rejected, 'diagnostics': describe(features)})
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.output_dir / 'features.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in detail_rows))
    print(json.dumps({'report': str(args.output_dir / 'report.json'),
                      'adapted': len(detail_rows), 'rejected': sum(len(d['rejected']) for d in result['datasets'])}))
    return 2 if any(d['rejected'] for d in result['datasets']) else 0


if __name__ == '__main__':
    raise SystemExit(main())

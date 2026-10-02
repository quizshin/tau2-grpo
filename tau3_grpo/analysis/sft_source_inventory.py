"""Read-only CPU inventory for the SFT redesign plan; no quality judge or replay."""
import argparse
import collections
import hashlib
import json
import pathlib
import statistics

from tau3_grpo.paths import AREAL_RAW_ROOT


def inventory(data_root):
    """Census source bytes and provided targets without inferring acceptance."""
    root = pathlib.Path(data_root)
    source = root / 'tau2_sft_train.jsonl'
    latest, indexes = {}, collections.defaultdict(list)
    target_positions = collections.defaultdict(set)
    quality = collections.defaultdict(set)
    total = airline = 0
    for line in source.open():
        r = json.loads(line)
        total += 1
        m = r.get('metadata', {})
        sid = m.get('source_dialog_id', '')
        if not sid.startswith('airline_dialog_'):
            continue
        airline += 1
        t = m['turn_index']
        indexes[sid].append(t)
        target_positions[sid].add(len(r['messages']))
        quality[sid].add((m.get('correct'), m.get('reward')))
        if sid not in latest or t > latest[sid]['metadata']['turn_index']:
            latest[sid] = r

    def messages(r):
        return r['messages'] + [r['answer']]

    def norm(m):
        return {k: m[k] for k in ('role', 'content', 'tool_calls', 'tool_call_id', 'name') if k in m}

    def digest(v):
        return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    prefixes = {}
    for sid, r in latest.items():
        ms = [norm(m) for m in messages(r)]
        prefixes[sid] = (ms, {})
    prefix_errors, answer_errors = [], []
    for line in source.open():
        r = json.loads(line)
        m = r.get('metadata', {})
        sid = m.get('source_dialog_id', '')
        if sid not in latest:
            continue
        full, cache = prefixes[sid]
        n = len(r['messages'])
        if n not in cache:
            cache[n] = digest(full[:n])
        if digest([norm(x) for x in r['messages']]) != cache[n]:
            prefix_errors.append([sid, m['turn_index']])
        if n >= len(full) or norm(r['answer']) != full[n]:
            answer_errors.append([sid, m['turn_index']])

    rl = []
    for line in (root / 'tau2_rl_train.jsonl').open():
        r = json.loads(line)
        ins = r.get('user_scenario', {}).get('instructions', {})
        if isinstance(ins, dict) and ins.get('domain') == 'airline':
            rl.append(r)
    reason_index = collections.defaultdict(list)
    for r in rl:
        reason_index[r['user_scenario']['instructions'].get('reason_for_call', '').strip().lower()].append(r['id'])

    records = []
    call_counts, coverage = collections.Counter(), collections.Counter()
    good_coverage, systems = collections.Counter(), collections.Counter()
    for sid, r in sorted(latest.items()):
        ms, meta = messages(r), r['metadata']
        calls = [c.get('function', c) for m in ms if m['role'] == 'assistant' for c in (m.get('tool_calls') or [])]
        names = collections.Counter(c.get('name') for c in calls)
        batches = [len(m.get('tool_calls') or []) for m in ms if m['role'] == 'assistant' and m.get('tool_calls')]
        good = quality[sid] == {(1, 1.0)}
        call_counts.update(names)
        coverage.update(names.keys())
        if good:
            good_coverage.update(names.keys())
        system = next((m.get('content', '') for m in ms if m['role'] == 'system'), '')
        systems[digest(system)] += 1
        idx = sorted(indexes[sid])
        unprovided = [dict(message_index=i, tools=[c.get('function', c).get('name') for c in (m.get('tool_calls') or [])], content_preview=(m.get('content') or '')[:200])
                      for i, m in enumerate(ms) if m['role']=='assistant' and i not in target_positions[sid]]
        records.append(dict(
            source_dialog_id=sid, reason_for_call=meta.get('reason_for_call'),
            seed_pattern_task_id=meta.get('seed_pattern_task_id'), source_labels=sorted(quality[sid]),
            source_eligible=good, source_row_count=len(idx), first_turn=idx[0], last_turn=idx[-1],
            missing_turn_indices=sorted(set(range(idx[0], idx[-1]+1))-set(idx)),
            assistant_turns=sum(m['role']=='assistant' for m in ms), tool_calls=len(calls),
            tool_counts=dict(names), multicall_turns=sum(n>1 for n in batches), max_batch=max(batches, default=0),
            content_plus_calls=sum(bool((m.get('content') or '').strip()) and bool(m.get('tool_calls')) for m in ms if m['role']=='assistant'),
            source_system_sha256=digest(system),
            provided_target_message_indices=sorted(target_positions[sid]),
            assistant_messages_without_source_target=unprovided,
            exact_reason_rl_ids=reason_index[meta.get('reason_for_call', '').strip().lower()],
            available_fields=sorted(r.keys()), metadata_fields=sorted(meta.keys()),
        ))
    summary = dict(
        source_sha256=hashlib.file_digest(source.open('rb'), 'sha256').hexdigest(),
        scope='Read-only source census, no semantic judge, no environment replay, no token rendering',
        all_sft_rows=total, airline_turn_rows=airline, airline_dialogues=len(records),
        source_label_pairs=dict(collections.Counter(str(sorted(v)) for v in quality.values())),
        all_turns_1_1=sum(r['source_eligible'] for r in records),
        unique_source_systems=len(systems),
        tool_call_counts=dict(call_counts), tool_dialogue_coverage=dict(coverage), eligible_tool_dialogue_coverage=dict(good_coverage),
        dialogues_with_multicall=sum(r['multicall_turns']>0 for r in records),
        multicall_assistant_turns=sum(r['multicall_turns'] for r in records),
        content_plus_calls_turns=sum(r['content_plus_calls'] for r in records),
        mean_tool_calls=statistics.mean(r['tool_calls'] for r in records),
        median_tool_calls=statistics.median(r['tool_calls'] for r in records),
        missing_index_dialogues=sum(bool(r['missing_turn_indices']) for r in records),
        initial_greetings_without_source_target=sum(any(m['message_index']==1 for m in r['assistant_messages_without_source_target']) for r in records),
        other_assistant_messages_without_source_target=sum(sum(m['message_index']!=1 for m in r['assistant_messages_without_source_target']) for r in records),
        missing_index_examples=[r for r in records if r['missing_turn_indices']][:3],
        duplicate_turn_index_dialogues=sum(len(v)!=len(set(v)) for v in indexes.values()),
        prefix_mismatch_rows=len(prefix_errors), prefix_mismatch_examples=prefix_errors[:10],
        answer_mismatch_rows=len(answer_errors), answer_mismatch_examples=answer_errors[:10],
        rl_airline_tasks=len(rl), exact_reason_join_dialogues=sum(bool(r['exact_reason_rl_ids']) for r in records),
        source_metadata_keys=sorted({k for r in records for k in r['metadata_fields']}),
        pattern_counts=dict(collections.Counter(r['seed_pattern_task_id'] for r in records)),
    )
    return dict(summary=summary, records=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=pathlib.Path, default=AREAL_RAW_ROOT)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Use a new source inventory output; preserve prior evidence')
    result = inventory(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()

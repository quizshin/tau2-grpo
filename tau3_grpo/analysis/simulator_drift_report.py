"""Evidence-preserving normalization and reporting; no API calls or rescoring."""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path

from tau3_grpo.analysis.simulator_drift import DRIFT, validate_review
from tau3_grpo.utils.hashing import sha256_file


def recover_packet(call):
    if isinstance(call.get('packet'), dict):
        return copy.deepcopy(call['packet']), []
    text = (call.get('raw_response') or {}).get('content', '').strip()
    try:
        return json.loads(text), ['parsed_saved_raw_json']
    except (ValueError, TypeError):
        pass
    # Some responses prematurely close after findings, then append metadata.
    # Recover only two disjoint, independently valid objects with known keys.
    try:
        first, end = json.JSONDecoder().raw_decode(text)
        tail = text[end:].strip().lstrip(',').strip()
        second = json.loads('{' + tail)
        if (set(first) <= {'coverage', 'findings'} and 'findings' in first
                and set(second) <= {'summary', 'uncertainties'} and not set(first) & set(second)):
            return {**first, **second}, ['merged_disjoint_findings_and_metadata_objects']
    except (ValueError, TypeError):
        pass
    return None, []


def normalize_review(packet, payload):
    packet = copy.deepcopy(packet)
    changes, rejected = [], []
    if not isinstance(packet, dict) or not isinstance(packet.get('findings'), list):
        return None, changes, [{'reason': 'No recoverable findings array'}]
    if not isinstance(packet.get('uncertainties'), list):
        packet['uncertainties'] = ['Judge omitted uncertainties; completeness not established.']
        packet['coverage'] = 'incomplete'
        changes.append('missing_uncertainties_marked_incomplete')
    if packet.get('coverage') not in {'complete', 'incomplete'}:
        packet['coverage'] = 'incomplete'
        changes.append('missing_coverage_marked_incomplete')
    clause_refs = {c['id']: c['source_refs'] for c in payload['contract_index']['clauses']}
    events = {e['event_id']: e for e in payload['events']}
    usable = []
    for index, original in enumerate(packet['findings']):
        f = copy.deepcopy(original)
        refs = f.get('scenario_refs', [])
        if isinstance(refs, list) and any(r in clause_refs for r in refs):
            f['scenario_refs'] = list(dict.fromkeys(r2 for r in refs for r2 in clause_refs.get(r, [r])))
            changes.append({'finding': index, 'kind': 'expanded_frozen_contract_refs',
                            'before': refs, 'after': f['scenario_refs']})
        if f.get('consequence') not in {'unknown', 'no_visible_change'} and not f.get('outcome_refs'):
            f['consequence'] = 'unknown'
            changes.append({'finding': index, 'kind': 'unsupported_consequence_downgraded_to_unknown'})
        message = events.get(f.get('event_id'), {})
        quote = f.get('user_quote')
        text = message.get('content') or ''
        if isinstance(quote, str) and quote and quote not in text:
            pattern = r'\s+'.join(re.escape(part) for part in quote.split())
            match = re.search(pattern, text) if pattern else None
            if match:
                f['user_quote'] = match.group()
                changes.append({'finding': index, 'kind': 'literal_quote_whitespace_only'})
        try:
            validate_review({'coverage': 'complete', 'findings': [f], 'uncertainties': []},
                            payload['events'], payload['scenario_source_lines'])
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            rejected.append({'finding_index': index, 'reason': str(exc), 'raw_finding': original})
        else:
            usable.append(f)
    packet['findings'] = usable
    if rejected:
        packet['coverage'] = 'incomplete'
    return packet, changes, rejected


def build_report(output):
    index = json.loads((output / 'private_case_index.json').read_text())
    records, counts = [], {}
    for item in index:
        rid = item['record_id']
        path = output / 'calls' / f'driftv4_{rid}_review.json'
        rec = {**item, 'status': 'not_processed', 'normalizations': [], 'quarantined': []}
        rec['review_source'] = 'primary'
        repair = output / 'repairs' / f'{rid}.json'
        if repair.exists():
            attempt = json.loads(repair.read_text())
            rec['repair_attempt_status'] = attempt['status']
            if attempt['status'] == 'validated_schema':
                rec['original_call_sha256'] = sha256_file(path)
                path = output / 'calls' / f'driftv4_{rid}_repairs1.json'
                rec['review_source'] = 'repairs1'
        if path.exists():
            call = json.loads(path.read_text())
            packet, parse_changes = recover_packet(call)
            rec['raw_call_sha256'] = sha256_file(path)
            if packet is not None:
                review, changes, rejected = normalize_review(packet, call['request'])
                rec.update(review=review, normalizations=parse_changes + changes, quarantined=rejected,
                           status='review_complete' if review and review['coverage'] == 'complete' else 'partial')
            else:
                rec['status'] = 'unusable_json'
        flags = (rec.get('review') or {}).get('findings', [])
        direct = [f for f in flags if f['support'] == 'direct']
        rec['direct_drift_candidate'] = any(f['kind'] in DRIFT for f in direct)
        rec['major_direct_drift_candidate'] = any(f['kind'] in DRIFT and f['severity'] == 'major' for f in direct)
        records.append(rec)
    for arm in ('base', 'sft1', 'sft3'):
        selected = [r for r in records if r['arm'] == arm]
        summary = dict(planned=len(selected), statuses=dict(Counter(r['status'] for r in selected)),
                       direct_drift_candidates=sum(r['direct_drift_candidate'] for r in selected),
                       major_direct_drift_candidates=sum(r['major_direct_drift_candidate'] for r in selected),
                       normalized_cases=sum(bool(r['normalizations']) for r in selected),
                       quarantined_findings=sum(len(r['quarantined']) for r in selected),
                       kinds=dict(Counter(k for r in selected for k in
                           {f['kind'] for f in (r.get('review') or {}).get('findings', []) if f['support'] == 'direct'})))
        summary['by_outcome'] = {}
        for outcome, success in (('success', True), ('failure', False)):
            sub = [r for r in selected if (r['reward'] >= 1 - 1e-6) == success]
            summary['by_outcome'][outcome] = {'planned': len(sub),
                'complete': sum(r['status'] == 'review_complete' for r in sub),
                'direct_drift_candidates': sum(r['direct_drift_candidate'] for r in sub)}
        counts[arm] = summary
    result = {'purpose': 'machine_candidate_census_not_true_drift_prevalence',
              'original_rewards_unchanged': True, 'arms': counts,
              'full_semantic_human_adjudication': False,
              'normalization_does_not_prove_semantic_correctness': True}
    (output / 'normalized_case_results.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    (output / 'normalized_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.output), ensure_ascii=False, indent=2))

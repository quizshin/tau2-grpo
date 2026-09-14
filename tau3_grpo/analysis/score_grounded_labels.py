"""Score small, assistant-reviewed literal witnesses; not semantic accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau3_grpo.algorithms.anchors.grounded import extract
from tau3_grpo.analysis.replay_decisions import visible_message
from tau3_grpo.utils.hashing import sha256_file, sha256_json


def score(labels):
    cache, rows = {}, []
    for label in labels['labels']:
        source = label['source']
        if source not in cache:
            if sha256_file(source) != labels['source_sha256'][source]:
                raise ValueError('Review source changed')
            cache[source] = [json.loads(line) for line in Path(source).open()]
        raw = cache[source][label['source_line'] - 1]
        if (raw['task_id'], raw['trial']) != (label['task_id'], label['trial']):
            raise ValueError('Review identity mismatch')
        prefix = [visible_message(m) for m in raw['simulation']['messages'][:label['prefix_end']]]
        if sha256_json(prefix) != label['prefix_sha256']:
            raise ValueError('Review prefix changed')
        evidence = label['evidence']; index = evidence['message_index']
        if not (0 <= index < len(prefix)):
            raise ValueError('Review cites outside prefix')
        text = prefix[index].get('content') or ''
        start, end = evidence['start'], evidence['end']
        if (not 0 <= start < end <= len(text) or text[start:end] != evidence['quote']
                or prefix[index]['role'] != evidence['role']):
            raise ValueError('Review evidence mismatch')
        report = extract(prefix)
        matched = any(c['kind'] == label['kind'] and c['value'] == label['value']
                      and c['evidence']['message_index'] == index for c in report['claims'])
        rows.append({'id': label['id'], 'evidence_valid': True, 'extracted': matched,
                     'kind': label['kind'], 'expected_value': label['value']})
    return {'labels': len(rows), 'evidence_valid': len(rows), 'extracted': sum(r['extracted'] for r in rows),
            'results': rows, 'scope': 'Purposively selected literal witnesses, not population recall, precision or semantic accuracy.',
            'independent_human_labels': False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--labels', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = score(json.loads(Path(args.labels).read_text()))
    with Path(args.output).open('x') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Inspect outcomes only AFTER reviewed labels and anchor predictions are frozen.

This cannot label pairs, change anchors, or infer causal action contributions.
"""
import argparse
import hashlib
import json
from pathlib import Path
from env_info.a800_20260912.audit_decision_pairs import digest


def inspect(fixture_path, audit_path, source):
    fixture=json.loads(fixture_path.read_text());audit=json.loads(audit_path.read_text())
    if digest(fixture)!=audit['fixture_sha256_canonical']:raise ValueError('Labels changed after prediction audit')
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_hash!=fixture['sources']['e2/trajectories.jsonl']:raise ValueError('Source mismatch')
    rows={}
    for line in source.open():
        if line.strip():
            row=json.loads(line);rows[(row['task_id'],row['trial'])]=row
    pairs=[]
    for pair in fixture['pairs']:
        if pair['kind']!='natural' or pair['relation']!='merge':continue
        a,b=[rows[(r['task_id'],r['trial'])] for r in pair['source_refs']]
        pairs.append({'pair_id':pair['id'],'task_id':a['task_id'],'trials':[a['trial'],b['trial']],
                      'terminal_rewards':[a['reward'],b['reward']],'discordant':a['reward']!=b['reward']})
    return {'scope':'post-label outcome inspection of correlated early-state pairs; not step advantages or causal benefit',
            'fixture_sha256_canonical':digest(fixture),'audit_sha256':hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            'source_sha256':source_hash,'used_for_labels_or_anchor_inputs':False,'pairs':pairs,
            'discordant_pairs':sum(p['discordant'] for p in pairs),
            'discordant_tasks':sorted({p['task_id'] for p in pairs if p['discordant']})}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=inspect(a.fixture,a.audit,a.source);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},ensure_ascii=False))

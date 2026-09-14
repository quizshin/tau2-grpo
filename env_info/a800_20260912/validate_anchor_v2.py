"""Compare live v1/v2 hooks on saved dialogues using a constant DB proxy.

This is a CPU wiring/partition audit, NOT recovered online anchor coverage.
No future actions, final DB, rewards or trial IDs enter the anchor input.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import time

from tau3_grpo.algorithms.anchors.evidence import decision_evidence
from tau3_grpo.envs.registry import SESSIONS, SessionEntry
from tau3_grpo.integrations.anchor_hook import current_anchor


def convert(raw):
    raw=dict(raw);raw['tool_calls']=[NS(**c) for c in raw.get('tool_calls') or []]
    return NS(**raw)


def validate(paths, versions=("v1", "v2")):
    source={};groups={version:defaultdict(list) for version in versions};counts=Counter();rows=[];seconds=Counter();examples=[]
    for path in paths:
        source[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as f:
            for line in f:
                if not line.strip():continue
                row=json.loads(line);counts['trajectories']+=1
                identity=f'{path.parent.name}:{row["task_id"]}:{row["trial"]}'
                prefix=[];session=NS(messages=prefix,task_id=row['task_id'],db_hash=lambda:'DB_PROXY_NOT_REPLAY')
                entry=SessionEntry(session=session);SESSIONS.register('audit-v2',entry);data=NS(request_id='audit-v2')
                turn=0
                try:
                    for message_index,raw in enumerate(row['simulation']['messages']):
                        msg=convert(raw)
                        if msg.role=='assistant':
                            ids={}
                            for version in groups:
                                entry.anchor_version=version;start=time.perf_counter();aid=current_anchor(data,'assistant');seconds[version]+=time.perf_counter()-start
                                ids[version]=aid;groups[version][(path.parent.name,aid)].append((identity,turn))
                            rows.append({'identity':identity,'assistant_step':turn,'message_index':message_index,**ids})
                            counts['assistant_decisions']+=1;counts['noninitial_decisions']+=int(turn>0);turn+=1
                        prefix.append(msg)
                        if msg.role in ('assistant', 'user') and msg.content:
                            from tau3_grpo.algorithms.anchors.semantic import normalize_utterance
                            kind=normalize_utterance(msg.role, str(msg.content))['kind']
                            counts['utterance/'+kind]+=1
                        if msg.role=='user' and path.parent.name=='e2' and (row['task_id'],row['trial'],message_index) in {('airline_1035',0,9),('airline_1060',1,13),('airline_1106',2,12)}:
                            ev=decision_evidence(prefix);examples.append({'task_id':row['task_id'],'trial':row['trial'],'message_index':message_index,
                                'user':msg.content,'reply_kind':ev.reply_kind,'payload':ev.payload()})
                finally:SESSIONS.pop('audit-v2')
    partition={}
    for version,buckets in groups.items():
        eligible=[entries for entries in buckets.values() if len({i for i,_ in entries})>=2]
        noninitial=sum(step>0 for entries in eligible for _,step in entries)
        partition[version]={'unique':len(buckets),'cross_trajectory_repeated_steps':sum(map(len,eligible)),
                            'cross_trajectory_noninitial_repeated_steps':noninitial,
                            'cross_trajectory_noninitial_fraction':noninitial/counts['noninitial_decisions'] if counts['noninitial_decisions'] else 0.,
                            'hook_seconds':seconds[version], 'mean_hook_ms':seconds[version]/counts['assistant_decisions']*1000}
    return {'scope':'saved selection dialogues; fixed DB proxy, group within arm/task over four trials; NOT online training coverage or nonzero signal',
            'counts':dict(counts),'partitions':partition,'reviewed_examples':examples,'source_sha256':source,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'decisions':rows,
            'models_called':False,'gpu_used':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--versions',nargs='+',default=['v1','v2']);a=p.parse_args()
    result=validate(a.input, versions=a.versions);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('decisions','reviewed_examples')},ensure_ascii=False))

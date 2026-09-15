"""Read-only audit of structured selection dialogues, without models or GPUs.

Missing intermediate DB hashes prevent reconstruction of actual training anchors.
The projection below is explicitly not an online anchor or coverage estimate.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

from tau3_grpo.algorithms.anchors.features import (
    confirmation_flags, known_info_mask, last_observation_type, policy_precondition_flags,
)
from env_info.a800_20260912.cpu_signal_validation import observation_fingerprint, stable_hash


def convert(raw):
    value=dict(raw)
    value['tool_calls']=[SimpleNamespace(**c) for c in value.get('tool_calls') or []]
    return SimpleNamespace(**value)


def audit(paths):
    sources={}; counts=Counter(); buckets=defaultdict(set); examples=defaultdict(list)
    for path in paths:
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        arm=path.parent.name
        with path.open() as stream:
            for line in stream:
                if not line.strip():continue
                row=json.loads(line); counts['trajectories']+=1
                messages=row['simulation']['messages'];prefix=[]; previously=False
                for turn,raw in enumerate(messages):
                    msg=convert(raw)
                    if msg.role=='assistant':
                        projection=stable_hash([row['task_id'], known_info_mask(prefix), confirmation_flags(prefix),
                                                policy_precondition_flags(prefix),str(last_observation_type(prefix))])
                        buckets[(arm,projection)].add(observation_fingerprint(prefix))
                        counts['assistant_decisions']+=1
                    prefix.append(msg)
                    confirmed='user_confirmed' in confirmation_flags(prefix)
                    if msg.role=='user':
                        text=msg.content or ''
                        if text.strip() == '###STOP###':
                            counts['terminal_control_messages']+=1
                            previously=confirmed
                            continue
                        counts['user_turns']+=1
                        single='user_confirmed' in confirmation_flags([msg])
                        counts['keyword_matching_user_turns']+=int(single)
                        item={'source':str(path),'task_id':row['task_id'],'trial':row.get('trial'),
                              'turn':turn,'user':text,'previous_assistant':next((x.content for x in reversed(prefix[:-1]) if x.role=='assistant' and x.content),None)}
                        if confirmed and not previously:
                            counts['first_confirmation_latches']+=1
                            if len(examples['first_latch'])<12:examples['first_latch'].append(item)
                        # Screening categories, NOT semantic ground truth. Keep context.
                        cues={
                            'negative_or_pause_keyword_hit': single and bool(re.search(r"\b(?:not|don't|no|wait|hold on|withdraw|stop)\b",text,re.I)),
                            'question_keyword_hit': single and '?' in text,
                            'identity_keyword_hit': single and bool(re.search(r'\b(?:my name|user id|reservation id|membership)\b',text,re.I)),
                            'latched_after_pause_cue': previously and bool(re.search(r"\b(?:don't proceed|do not proceed|withdraw|wait|hold on|stop)\b",text,re.I)),
                        }
                        for key,hit in cues.items():
                            if hit:
                                counts[key]+=1
                                if len(examples[key])<12:examples[key].append(item)
                    previously=confirmed
    refined=sum(len(v)>1 for v in buckets.values())
    return {'sources_sha256':sources,'counts':dict(counts),'screening_examples':dict(examples),
            'read_projection':{'buckets':len(buckets),'buckets_with_multiple_read_ledgers':refined,
                               'ledger_partitions':sum(map(len,buckets.values()))},
            'scope':'structured selection dialogues, not training replay; cue counts require contextual review; DB omitted from projection so these are not confirmed full-anchor collisions or online coverage',
            'cuda_used':False,'models_called':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.input);result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='screening_examples'},ensure_ascii=False))

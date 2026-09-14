"""Strict CPU replay of recorded selection tools into fresh official Airline DBs.

Only observed assistant/user text is replayed; no user simulator is constructed.
A decision is certified only through the last verified tool response, under the
pinned local environment implementation. This does not recover hidden user state.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
import uuid

import yaml

from tau3_grpo.algorithms.anchors.evidence import validate_version
from tau3_grpo.data.manifest import read_manifest
from tau3_grpo.data.schema import ArealTaskRecord, DataSource
from tau3_grpo.envs.adapter import build_environment, load_flight_db
from tau3_grpo.envs.registry import SESSIONS, SessionEntry
from tau3_grpo.envs.tau2_bridge import message_models
from tau3_grpo.integrations.anchor_hook import current_anchor
from tau3_grpo.paths import CODE_ROOT
from tau3_grpo.utils.hashing import sha256_file, sha256_json, sha256_text


def normalized_content(value: Any, comparison: str) -> str:
    if comparison == 'exact_text':
        return value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
    if comparison != 'json':
        raise ValueError(f'unsupported comparison: {comparison}')
    if isinstance(value,str):
        try:value=json.loads(value)
        except (ValueError,TypeError):return 'text:'+value
    # Ignores object key order / whitespace only; array order, text, bool and
    # number representations remain distinct. No fuzzy matching of observations.
    return 'json:'+json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def visible_message(raw):
    return {k:deepcopy(raw[k]) for k in ('role','content','tool_calls','id','tool_call_id','error','requestor') if k in raw}


def as_object(raw):
    obj=deepcopy(raw);obj['tool_calls']=[NS(**c) for c in obj.get('tool_calls') or []]
    return NS(**obj)


def replay_messages(environment, messages, *, task_id, versions=('v1','v2','v3'), comparison='json', expected_policy=None):
    """Replay one trajectory. Stop certification immediately on any mismatch.

    Snapshots preceding a later mismatch remain valid verified prefixes. No
    state after that mismatch is exposed as certified. All calls in a batch
    execute in list order; malformed/missing/out-of-order responses abort.
    """
    for version in versions:validate_version(version)
    normalized_content('',comparison)
    prefix=[];pending=deque();snapshots=[];events=[];used_ids=set();failure=None
    policy_hash=sha256_text(environment.get_policy())
    if expected_policy is not None and sha256_text(expected_policy)!=policy_hash:
        return {'status':'unverified','failure':{'kind':'policy_mismatch','message_index':0},'snapshots':[],'events':[]}
    models=message_models()
    def fail(kind,index,**extra):return {'kind':kind,'message_index':index,**extra}
    for index,raw in enumerate(messages):
        role=raw.get('role')
        if pending and role!='tool':
            failure=fail('missing_tool_response',index);break
        if role=='assistant':
            db_hash=environment.get_db_hash()
            if db_hash is None:
                failure=fail('missing_db_hash',index);break
            session=NS(task_id=task_id,messages=[as_object(m) for m in prefix],db_hash=lambda:db_hash,policy=environment.get_policy)
            request='replay-'+uuid.uuid4().hex;entry=SessionEntry(session=session);anchors={};diagnostics={}
            SESSIONS.register(request,entry)
            try:
                for version in versions:
                    entry.anchor_version=version
                    anchors[version]=current_anchor(NS(request_id=request),'assistant')
                    if version=='v4':diagnostics[version]=deepcopy(entry.decision_diagnostics)
            finally:SESSIONS.pop(request)
            snapshots.append({'message_index':index,'assistant_step':len(snapshots),'db_hash':db_hash,
                              'policy_hash':policy_hash,'prefix_hash':sha256_json(prefix),'anchors':anchors,
                              'verified_tool_responses':len(events),'certified_prefix':True,'decision_diagnostics':diagnostics})
            calls=raw.get('tool_calls') or []
            for call in calls:
                if not call.get('id') or call['id'] in used_ids:
                    failure=fail('duplicate_or_missing_call_id',index);break
                if call.get('requestor','assistant')!='assistant':
                    failure=fail('unsupported_tool_requestor',index);break
                used_ids.add(call['id']);pending.append(call)
            if failure:break
        elif role=='tool':
            call_id=raw.get('id') or raw.get('tool_call_id')
            if not pending or pending[0]['id']!=call_id:
                failure=fail('unexpected_tool_response_order',index,call_id=call_id);break
            call=pending.popleft()
            if raw.get('requestor','assistant')!='assistant':
                failure=fail('unsupported_tool_requestor',index);break
            try:
                tool_call=models['ToolCall'].model_validate(call)
                before=environment.get_db_hash()
                response=environment.get_response(tool_call)
                left=normalized_content(response.content,comparison)
                right=normalized_content(raw.get('content'),comparison)
            except Exception as exc:
                failure=fail('replay_exception',index,exception_type=type(exc).__name__);break
            if response.id!=call_id or bool(response.error)!=bool(raw.get('error',False)) or left!=right:
                failure=fail('tool_result_mismatch',index,call_id=call_id,tool_name=call['name'],
                             actual_error=bool(response.error),recorded_error=bool(raw.get('error',False)),
                             actual_content_hash=sha256_text(left),recorded_content_hash=sha256_text(right));break
            events.append({'message_index':index,'tool_name':call['name'],'call_id':call_id,
                           'db_before':before,'db_after':environment.get_db_hash(),'error':bool(response.error)})
        elif role=='user':
            if raw.get('tool_calls'):
                failure=fail('unsupported_user_tools',index);break
        elif role!='system':
            failure=fail('unsupported_message_role',index,role=role);break
        prefix.append(visible_message(raw))
    if failure is None and pending:failure=fail('missing_tool_response',len(messages))
    return {'status':'verified' if failure is None else 'partial' if snapshots else 'unverified',
            'failure':failure,'snapshots':snapshots,'events':events}


def replay_entry(row, entry, db_root, *, versions, comparison):
    if entry.source != DataSource.AREAL_TAU2_AIRLINE or entry.split!='selection':
        raise ValueError('Replay diagnostic only accepts frozen AReaL selection tasks')
    record=ArealTaskRecord.model_validate(entry.task)
    if record.id!=entry.task_id or row.get('task_id')!=record.id or record.fingerprint!=entry.task_hash:
        raise ValueError('Task identity or pinned record fingerprint mismatch')
    if record.initial_state:
        raise ValueError('Nonempty initial_state requires explicit initialization support')
    if record.db_path!=entry.db_path:raise ValueError('Manifest DB path mismatch')
    path=record.resolve_db_path(db_root)
    if not entry.db_hash or sha256_file(path)!=entry.db_hash:raise ValueError('Pinned DB file hash mismatch')
    simulation=row.get('simulation',{})
    if simulation.get('task_id')!=record.id:raise ValueError('Simulation task mismatch')
    if simulation.get('mode')!='half_duplex':raise ValueError('Only text half-duplex records supported')
    if not isinstance(simulation.get('policy'),str):raise ValueError('Recorded policy required')
    env=build_environment(load_flight_db(path))
    env.set_state(initialization_data=None,initialization_actions=None,message_history=[])
    result=replay_messages(env,simulation['messages'],task_id=record.id,versions=versions,
                           comparison=comparison,expected_policy=simulation['policy'])
    return {'task_id':record.id,'trial':row.get('trial'),'db_file_hash':entry.db_hash,
            'task_hash':entry.task_hash,**result}


def resolve(path):
    path=Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (CODE_ROOT/path).resolve()


def run(config, inputs, output):
    if not config.get('enabled',False):return {'enabled':False,'models_called':False}
    versions=config.get('anchor_versions',['v1','v2','v3'])
    if not versions or len(set(versions))!=len(versions):raise ValueError('Unique nonempty anchor_versions required')
    for version in versions:validate_version(version)
    comparison=config.get('comparison','json');normalized_content('',comparison)
    manifest=resolve(config['manifest']);db_root=resolve(config['db_root'])
    entries={e.task_id:e for e in read_manifest(manifest)}
    selected=set(config.get('task_ids') or entries)
    if not selected.issubset(entries):raise ValueError('Unknown selected task')
    output=resolve(output)
    if output.exists():raise FileExistsError('Use a new output directory; replay evidence is immutable')
    output.mkdir(parents=True)
    counts=Counter();grouped={v:{} for v in versions};sources={};seen=set();errors=[]
    with (output/'trajectories.jsonl').open('w') as stream:
        for source in inputs:
            source=resolve(source);sources[str(source)]=sha256_file(source)
            with source.open() as handle:
                for line_no,line in enumerate(handle,1):
                    if not line.strip():continue
                    row=json.loads(line)
                    if row.get('task_id') not in selected:continue
                    identity=(str(source),row['task_id'],row.get('trial'))
                    if identity in seen:raise ValueError('Duplicate trajectory identity')
                    seen.add(identity)
                    try:r=replay_entry(row,entries[row['task_id']],db_root,versions=versions,comparison=comparison)
                    except Exception as exc:
                        r={'task_id':row['task_id'],'trial':row.get('trial'),'status':'unverified',
                           'failure':{'kind':'initialization_or_replay_error','exception_type':type(exc).__name__,'detail':str(exc)},'snapshots':[],'events':[]}
                    r['source']=str(source);r['source_line']=line_no
                    counts['trajectories']+=1;counts[r['status']]+=1;counts['verified_tool_responses']+=len(r['events'])
                    if r['failure']:counts['failure/'+r['failure']['kind']]+=1;errors.append({k:r[k] for k in ['task_id','trial','source','failure']})
                    for s in r['snapshots']:
                        counts['certified_decisions']+=1;counts['certified_noninitial_decisions']+=int(s['assistant_step']>0)
                        for v in versions:
                            if s['anchors'][v] in (None, 'abstain:v4'):
                                counts['abstained/'+v]+=1
                                for reason in s.get('decision_diagnostics',{}).get(v,{}).get('reasons',[]):
                                    counts['abstention_reason/'+v+'/'+reason]+=1
                                continue
                            counts['emitted/'+v]+=1
                            # Never mix experimental arms or tasks. This analysis groups
                            # evaluation trials, NOT original training rollout groups.
                            key=(str(source),r['task_id'],s['anchors'][v])
                            grouped[v].setdefault(key,[]).append((identity,s['assistant_step']))
                    stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
    if not counts['trajectories']:raise ValueError('No selected trajectories found')
    partitions={}
    for v,buckets in grouped.items():
        eligible=[xs for xs in buckets.values() if len({ident for ident,_ in xs})>=2]
        n=sum(step>0 for xs in eligible for _,step in xs)
        partitions[v]={'cross_trial_noninitial_repeated_decisions':n,
                       'fraction_of_certified_noninitial':n/counts['certified_noninitial_decisions'] if counts['certified_noninitial_decisions'] else 0}
    sources[str(manifest)]=sha256_file(manifest)
    # Record local implementation provenance, including domain tool code.
    code_hashes={str(p.relative_to(CODE_ROOT)):sha256_file(p) for base in ['tau3_grpo/algorithms/anchors','tau3_grpo/analysis','tau2-bench/src/tau2/domains/airline'] for p in (CODE_ROOT/base).glob('*.py')}
    for rel in ['tau3_grpo/integrations/anchor_hook.py','tau3_grpo/envs/adapter.py','tau2-bench/src/tau2/environment/environment.py']:
        code_hashes[rel]=sha256_file(CODE_ROOT/rel)
    result={'enabled':True,'config':config,'counts':dict(counts),'partitions':partitions,'failures':errors,
            'source_sha256':sources,'code_sha256':code_hashes,'models_called':False,'gpu_used':False,
            'scope':'tool-verified pre-action DB and visible prefixes under local code; not hidden simulator-state replay, semantic-equivalence labels, training coverage, or advantage estimates'}
    (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    from loguru import logger
    logger.disable('tau2')  # Avoid dumping whole DB/tool payloads during analysis.
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--input',action='append',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=run(yaml.safe_load(resolve(a.config).read_text()),a.input,a.output)
    print(json.dumps({k:v for k,v in result.items() if k in ('enabled','counts','partitions','models_called','scope')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

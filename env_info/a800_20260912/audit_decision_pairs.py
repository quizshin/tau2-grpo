"""CPU-only pairwise diagnostic through the production anchor hook.

No policy/user models, source outcomes or future messages are consulted.
A merge prediction is anchor equality, not an assertion of true Markov equality.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

from tau3_grpo.algorithms.anchors.encoder import AnchorMode
from tau3_grpo.algorithms.anchors.evidence import decision_evidence
from tau3_grpo.envs.registry import SESSIONS, SessionEntry
from tau3_grpo.integrations.anchor_hook import current_anchor


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def messages(raw):
    result=[]
    for item in raw:
        item=deepcopy(item)
        item['tool_calls']=[NS(**c) for c in item.get('tool_calls') or []]
        result.append(NS(**item))
    return result


def anchor(state,version):
    session=NS(task_id=state['task_id'],db_hash=lambda:state['db_hash'],messages=messages(state['messages']))
    entry=SessionEntry(session=session,anchor_version='v2' if version=='db_only' else version,
                       anchor_mode=AnchorMode.DB_HASH_ONLY if version=='db_only' else AnchorMode.STRUCTURED)
    # A unique request ID per call, preserving any unrelated registry entries.
    import uuid
    request='pair-audit-'+uuid.uuid4().hex
    SESSIONS.register(request,entry)
    try:return current_anchor(NS(request_id=request),'assistant')
    finally:SESSIONS.pop(request)


def validate(fixture):
    seen=set()
    for pair in fixture['pairs']:
        if pair['id'] in seen:raise ValueError('duplicate pair id')
        seen.add(pair['id'])
        if pair['relation'] not in ('merge','separate','uncertain'):raise ValueError('invalid relation')
        if not pair['rationale']:raise ValueError('missing reviewed rationale')
        for side in ('a','b'):
            s=pair[side]
            if set(s)!={'task_id','db_hash','remaining_assistant_turns','messages'}:raise ValueError('unexpected state field (including potential outcome leakage)')
            if not s['messages']:raise ValueError('empty decision prefix')


def audit(fixture):
    validate(fixture)
    versions=('db_only','v1','v2','v3')
    counts=defaultdict(lambda:defaultdict(Counter));rows=[]
    for pair in fixture['pairs']:
        item={k:deepcopy(pair[k]) for k in ('id','kind','relation','rationale','source_refs')}
        item['prediction']={};item['anchors']={};item['evidence_differences']={}
        for version in versions:
            a,b=(anchor(pair[side],version) for side in ('a','b'))
            predicted='merge' if a==b else 'separate'
            item['prediction'][version]=predicted;item['anchors'][version]=[a,b]
            label=pair['relation'];c=counts[pair['kind']][version]
            if label=='uncertain':
                c['uncertain_'+predicted]+=1
            else:
                c['label_'+label]+=1
                c['correct' if predicted==label else ('false_merge' if predicted=='merge' else 'false_split')]+=1
        for version in ('v2','v3'):
            a,b=(decision_evidence(messages(pair[side]['messages']),version=version).payload() for side in ('a','b'))
            item['evidence_differences'][version]=[key for key in a if a[key]!=b[key]]
        item['remaining_budget_equal']=pair['a']['remaining_assistant_turns']==pair['b']['remaining_assistant_turns']
        rows.append(item)
    return {'fixture_sha256_canonical':digest(fixture),'annotation':fixture['annotation'],
            'scope':'purposive state-pair diagnostic, controlled matched DB; NOT model success or population accuracy',
            'models_called':False,'gpu_used':False,'counts':{k:{v:dict(c) for v,c in vs.items()} for k,vs in counts.items()},'pairs':rows}


def report(result):
    lines=['# 决策状态成对无卡审计（2026-09-14）','',result['scope'],'',
           '标签由 Codex 根据已见对话逐条审阅，未使用 reward 或后续动作；不是独立人工标注或随机测试集。',
           '自然样例均来自同任务不同 trial 的首次用户发言；真实前缀改写是控制反例，不能冒充自然出现的另一条轨迹。',
           '后续真实前缀只含读工具，仍按匹配 DB 假设检查；未恢复中间 DB 或用户模拟器隐藏状态。','',
           '|样例类别|方法|应合并|应分开|误合并|误拆分|不确定样例被合并/分开|','|---|---|---:|---:|---:|---:|---:|']
    for kind,methods in result['counts'].items():
        for version,c in methods.items():
            lines.append(f"|{kind}|{version}|{c.get('label_merge',0)}|{c.get('label_separate',0)}|{c.get('false_merge',0)}|{c.get('false_split',0)}|{c.get('uncertain_merge',0)}/{c.get('uncertain_separate',0)}|")
    lines+=['','## 逐对审阅与预测','', 'M=合并，S=分开；uncertain 不计对错。external_budget 独立检查有限时域边界，当前 hook 不读取预算。','',
            '|编号|审阅关系|DB-only/v1/v2/v3|理由|','|---|---|---|---|']
    for p in result['pairs']:
        pred='/'.join('M' if p['prediction'][v]=='merge' else 'S' for v in ('db_only','v1','v2','v3'))
        lines.append(f"|{p['id']}|{p['relation']}|{pred}|{p['rationale']}|")
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(json.loads(a.fixture.read_text()))
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),a.fixture,Path('tau3_grpo/algorithms/anchors/semantic.py'),Path('tau3_grpo/algorithms/anchors/evidence.py'),Path('tau3_grpo/integrations/anchor_hook.py')]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    a.output.with_suffix('.md').write_text(report(result))
    print(json.dumps(result['counts'],ensure_ascii=False,indent=2))

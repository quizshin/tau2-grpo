"""Render a traceable posthoc audit without modifying any official reward."""
import argparse
import copy
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

DIMENSIONS = {'intent': '意图与约束', 'evidence_arguments': '证据与参数',
              'action_compliance': '操作与政策', 'completion': '任务完成',
              'termination_efficiency': '终止与效率'}
CATEGORIES = {'agent_decision': '模型决策问题', 'simulator_drift': '模拟用户偏移',
              'scoring_conflict': '评分口径冲突', 'execution_limit': '执行上限',
              'infrastructure': '基础设施', 'unresolved': '待定'}
# Label normalization only: these mappings do not validate the finding's substance.
ALIASES = {'policy_compliance': 'action_compliance', 'intent_understanding': 'intent',
           'calculation': 'evidence_arguments', 'tool_usage': 'evidence_arguments',
           'payment_handling': 'evidence_arguments', 'confirmation_protocol': 'action_compliance',
           'user_confirmation': 'action_compliance', 'no_transfer': 'action_compliance',
           'intent_fulfillment': 'completion', 'task_completion': 'completion',
           'refund_calculation': 'evidence_arguments', 'conversation_quality': 'termination_efficiency',
           'compensation_calculation': 'evidence_arguments', 'confirmation_handling': 'action_compliance'}
GUIDE = '''这是 selection60 开发集的事后诊断，不是新测试集或官方重评分。
完整覆盖 Base 与 SFT 1 epoch 的 228 条失败，另有 12 条成功盲审对照；未逐条判读其余成功轨迹，不能据此计算全量能力正确率。
DeepSeek 输出仅为机器候选意见。schema 合格不代表语义正确，标签不是互斥原因，也不是因果证明。
原始检查表在轨迹判读前冻结；下面的适用规则和证据复核是事后补充，均单独留痕。
通用适用规则：只按行动前可见信息判断助手；隐藏 scenario 用于核查模拟用户，不能要求助手预知。
scenario 的 NO TRANSFER 是用户行为指令，不能覆盖助手按政策转人工的权限；合法改签、取消重订也不能一概禁止。
免费行李额度不等于已登记件数；Gold 商务舱每乘客 4 件免费。后续用户接受错误前提不能证明此前陈述正确。
多工具调用不自动算错；工具返回成功不等于任务成功；不要求照抄 gold 工具路径或每次额外读取确认。
数据库差异表示与唯一参考终态不一致，本身不能证明助手错误。执行上限是终止事实，不能自动解释根因。'''


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def normalize(packet, events):
    packet = copy.deepcopy(packet)
    roles = {e['event_id']: e['role'] for e in events}
    changes, rejected, findings = [], [], []
    for finding in packet.get('findings', []):
        old = finding.get('dimension')
        if old in ALIASES:
            finding['dimension'] = ALIASES[old]
            changes.append({'field': 'dimension', 'from': old, 'to': ALIASES[old],
                            'event_id': finding.get('event_id')})
        role = roles.get(finding.get('event_id'))
        refs = finding.get('evidence_refs', [])
        reason = None
        if finding.get('dimension') not in DIMENSIONS:
            reason = '未知能力标签，未猜测映射'
        elif not refs or not set(refs) <= roles.keys():
            reason = '缺失或无效证据引用'
        elif finding.get('category') == 'agent_decision' and role != 'assistant':
            reason = '决策事件不是 assistant，未猜测移到相邻事件'
        elif finding.get('category') == 'simulator_drift' and role != 'user':
            reason = '模拟用户事件不是 user'
        if reason:
            rejected.append({'finding': finding, 'reason': reason})
        else:
            findings.append(finding)
    packet['findings'] = findings
    return packet, changes, rejected


def outcome_tags(replay):
    tags = set()
    for d in replay.get('differences', []):
        path = d['path']
        if path.endswith('/passengers'):
            tags.add('乘客顺序差异' if d['kind'] == 'list_order_only' else '乘客内容差异')
        elif path.endswith('/payment_history') or '/payment_methods/' in path:
            tags.add('支付／退款／证书终态差异')
        elif path.endswith('/flights'):
            tags.add('航班／日期／舱位终态差异')
        elif 'baggages' in path:
            tags.add('行李终态差异')
        elif path.endswith('/status'):
            tags.add('预订状态差异')
        elif path.endswith('/cabin'):
            tags.add('航班／日期／舱位终态差异')
        else:
            tags.add('其他数据库差异')
    return sorted(tags)


def render(root, output):
    manifest = load(output / 'manifest.json')
    tasks = [json.loads(l) for l in (root / 'data/manifests/areal_airline_selection_seed42.jsonl').read_text().splitlines()]
    index = load(output / 'private_case_index.json')
    overrides = load(output / 'evidence_overrides.json')
    by_key = {(x['arm'], x['task_id'], x['trial']): x for x in overrides['cases']}
    rows = {}
    sources = {'base': 'base_selection_new_20260925/eval', 'sft1': 'sft_staged_v2_A100/seed42/eval'}
    input_hashes = {}
    for arm, rel in sources.items():
        path = root / 'results/runs' / rel / 'trajectories.jsonl'
        input_hashes[arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows[arm, row['task_id'], row['trial']] = row
    assert len(tasks) == 60 and len(rows) == 480 and len(index) == 240
    cases = [{'arm': x['arm'], 'row': rows[x['arm'], x['task_id'], x['trial']],
              'record_id': x['record_id']} for x in index]
    digest = hashlib.sha256(json.dumps({'tasks': tasks, 'cases': cases}, ensure_ascii=False,
                                      sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert digest == manifest['input_hash'], 'Audit input identity changed'
    frozen = load(output / 'rubric_freeze.json')
    assert all(hashlib.sha256((output / 'rubrics' / f'{t}.json').read_bytes()).hexdigest() == v for t, v in frozen.items())
    records = []
    for meta in index:
        rid = meta['record_id']; row = rows[meta['arm'], meta['task_id'], meta['trial']]
        call = load(output / 'calls' / f'sel60v1_{rid}_review.json')
        events = call['request']['events']
        normalized, changes, rejected = normalize(call['packet'], events)
        required = {c['id'] for c in load(output / 'rubrics' / f"{meta['task_id']}.json")['rubric']['checks']}
        assert len(normalized['checks']) == len(required)
        assert {c['id'] for c in normalized['checks']} == required
        event_ids = {e['event_id'] for e in events}
        assert all(set(c['evidence_refs']) <= event_ids for c in normalized['checks'])
        replay = load(output / 'replay' / f'{rid}.json')
        override = by_key.get((meta['arm'], meta['task_id'], meta['trial']), {})
        findings = normalized['findings']
        candidates = sorted({f['category'] for f in findings})
        if not candidates:
            candidates = ['unresolved']
        errors = [e['event_id'] for e in events if e['role'] == 'tool' and e.get('error')]
        termination = str(row['termination_reason'])
        limit = any(s in termination.lower() for s in ('max_steps', 'context', 'too_many', 'timeout'))
        evidence_tags = list(override.get('categories', []))
        if limit:
            evidence_tags.append('execution_limit')
        diffs = replay.get('differences', [])
        if diffs and all(d['kind'] == 'list_order_only' and d['path'].endswith('/passengers') for d in diffs):
            evidence_tags.append('scoring_conflict')
            override = copy.deepcopy(override)
            override.setdefault('notes', []).append('严格回放确认唯一终态差异为乘客列表顺序；姓名和生日集合完全一致。评分对顺序敏感，是否接受等价顺序需冻结新协议，当前分数不变。')
        record = dict(meta, is_control=meta['reward'] >= 1 - 1e-6,
                      official_reward_info=row['simulation'].get('reward_info'),
                      termination_reason=termination, tool_error_events=errors,
                      machine_categories=candidates, evidence_categories=sorted(set(evidence_tags)),
                      diagnosis_status=('evidence_spot_checked' if override else 'machine_candidate_not_semantically_verified'),
                      outcome_tags=outcome_tags(replay), review=normalized,
                      label_normalizations=changes, quarantined_findings=rejected,
                      evidence_override=override, replay=replay,
                      raw_review_status=load(output / 'reviews' / f'{rid}.json')['status'])
        records.append(record)
    failures = [r for r in records if not r['is_control']]
    assert len(failures) == 228 and Counter(r['arm'] for r in failures) == {'base': 109, 'sft1': 119}
    expected = {(a,t,tr) for (a,t,tr),r in rows.items() if r['reward'] < 1-1e-6}
    assert {(r['arm'],r['task_id'],r['trial']) for r in failures} == expected
    stats = {}
    tool_coverage = {}
    for task in tasks:
        names = {a['name'] for a in task['task']['evaluation_criteria'].get('actions', [])}
        for name in names:
            tool_coverage.setdefault(name, {'gold_task_count': 0, 'base_task_count': 0, 'sft1_task_count': 0})['gold_task_count'] += 1
    for arm in sources:
        seen = defaultdict(set)
        for (a, tid, _), row in rows.items():
            if a != arm:
                continue
            for message in row['simulation']['messages']:
                for call in message.get('tool_calls') or []:
                    seen[call['name']].add(tid)
        for name, ids in seen.items():
            tool_coverage.setdefault(name, {'gold_task_count': 0, 'base_task_count': 0, 'sft1_task_count': 0})[arm+'_task_count'] = len(ids)
    for arm in sources:
        fs = [r for r in failures if r['arm'] == arm]
        stats[arm] = {'failures': len(fs), 'machine_candidate_categories': dict(Counter(c for r in fs for c in r['machine_categories'])),
                      'evidence_categories': dict(Counter(c for r in fs for c in r['evidence_categories'])),
                      'outcome_tags': dict(Counter(c for r in fs for c in r['outcome_tags'])),
                      'tool_error_trajectories': sum(bool(r['tool_error_events']) for r in fs)}
    summary = {'scope': 'posthoc_diagnostic_not_rescoring', 'tasks': 60, 'failures': 228, 'controls': 12,
               'all_evaluation_records': 480, 'arms': stats,
               'normalized_cases': sum(bool(r['label_normalizations']) for r in records),
               'quarantined_cases': sum(bool(r['quarantined_findings']) for r in records),
               'quarantined_findings': sum(len(r['quarantined_findings']) for r in records),
               'replay_status': dict(Counter(r['replay']['status'] for r in records)),
               'saved_db_disagreements': sum(r['replay'].get('agrees_with_saved_db_check') is False for r in records),
               'scoring_checks_absent': sum(r['replay'].get('saved_db_match') is None for r in records),
               'input_sha256': input_hashes, 'audit_manifest': manifest,
               'tool_task_coverage': tool_coverage,
               'api': load(output / 'summary.json'),
               'report_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    save(output / 'report_summary.json', summary)
    (output / 'failure_classification.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in failures))
    save(output / 'successful_controls.json', [r for r in records if r['is_control']])
    grouped = defaultdict(list)
    for r in records:
        grouped[r['task_id']].append(r)
    for r in records:
        rid = r['record_id']; call = load(output / 'calls' / f'sel60v1_{rid}_review.json')
        lines = [f"# {r['task_id']} · {r['arm']} · trial {r['trial']}",
                 f"官方 reward：{r['reward']}；终止：{r['termination_reason']}；成功对照：{r['is_control']}",
                 '机器候选分类：'+ '、'.join(CATEGORIES[c] for c in r['machine_categories']),
                 '证据复核／终止事实：'+ '、'.join(CATEGORIES[c] for c in r['evidence_categories']),
                 '## 证据复核（优先于下方机器初判）',
                 json.dumps(r['evidence_override'], ensure_ascii=False, indent=2),
                 '## 客观终态差异（gold 不等于唯一合法解）',
                 '```json\n'+json.dumps(r['replay'],ensure_ascii=False,indent=2)+'\n```',
                 '## 机器逐项意见：未经逐条语义确认',
                 '```json\n'+json.dumps(r['review'],ensure_ascii=False,indent=2)+'\n```',
                 '## 标签规范化与隔离意见',
                 '```json\n'+json.dumps({'normalizations':r['label_normalizations'],'quarantined':r['quarantined_findings']},ensure_ascii=False,indent=2)+'\n```',
                 '## 可见消息证据']
        for e in call['request']['events']:
            lines.extend([f"### {e['event_id']} · {e['role']}", '```json\n'+json.dumps(e,ensure_ascii=False,indent=2)+'\n```'])
        path = output / 'cases' / f'{rid}.md'; path.parent.mkdir(exist_ok=True)
        path.write_text('\n\n'.join(lines)+'\n')
    task_rows, cards, task_machine = [], [], []
    for task in sorted(tasks, key=lambda t: int(t['task_id'].split('_')[-1])):
        tid = task['task_id']; rubric = load(output/'rubrics'/f'{tid}.json')['rubric']
        counts = {arm: sum(rows[arm,tid,tr]['reward'] >= 1-1e-6 for tr in range(4)) for arm in sources}
        tools = sorted({a['name'] for a in task['task']['evaluation_criteria'].get('actions',[])})
        checks = rubric['checks']
        notes = ['所有条目必须结合总说明中的适用规则；不能把私有用户指令升格为助手政策。']
        for check in checks:
            if any(s in check['criterion'] for s in ('转人工','转移','取消并重订')):
                notes.append(check['id']+'：需特别核对助手政策及用户私有指令的边界，不可一概禁止合法操作或转人工。')
        task_record = {'task_id': tid, 'goal': rubric['goal'], 'checks': checks,
                       'risks': rubric['risks'], 'applicability_notes': notes,
                       'reference_tool_names_not_mandatory': tools, 'official_successes_out_of_4': counts}
        task_machine.append(task_record)
        lines = [f'# {tid}',rubric['goal'],f"官方成功次数：Base {counts['base']}/4；SFT 1 epoch {counts['sft1']}/4。",
                 '## 适用边界', '\n'.join('- '+n for n in notes),
                 '## 逐题检查表', '|项|能力|检查内容|适用条件|所需证据|依据|\n|---|---|---|---|---|---|']
        for c in checks:
            cells = [c['id'], DIMENSIONS[c['dimension']],c['criterion'],c['condition'],c['evidence_needed'],','.join(c['source_refs'])]
            lines[-1] += '\n|'+'|'.join(x.replace('|','／').replace('\n',' ') for x in cells)+'|'
        lines += ['## 任务歧义候选', json.dumps(rubric['risks'],ensure_ascii=False,indent=2),
                  '## 参考工具（仅覆盖统计，不是必需调用清单）', ', '.join(tools),
                  '## 原始任务场景', '```json\n'+json.dumps(task['task']['user_scenario'],ensure_ascii=False,indent=2)+'\n```',
                  '## 失败及成功对照']
        for r in sorted(grouped[tid],key=lambda x:(x['arm'],x['trial'])):
            labels = '、'.join(CATEGORIES[c] for c in r['machine_categories'])
            lines.append(f"- [{r['arm']} trial {r['trial']} · reward {r['reward']}](../cases/{r['record_id']}.md)：机器候选 {labels}；证据复核 {r['diagnosis_status']}。")
        path=output/'tasks'/f'{tid}.md';path.parent.mkdir(exist_ok=True);path.write_text('\n\n'.join(lines)+'\n')
        task_rows.append(f"|[{tid}](tasks/{tid}.md)|{counts['base']}/4|{counts['sft1']}/4|{len(checks)}|{len(grouped[tid])}|")
        body = '<h3>逐题检查表</h3><table><tr><th>项</th><th>能力</th><th>标准</th><th>条件与证据</th></tr>'
        for c in checks:
            body += '<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in [c['id'],DIMENSIONS[c['dimension']],c['criterion'],c['condition']+' '+c['evidence_needed']])+'</tr>'
        body += '</table><p>'+html.escape(' '.join(notes))+'</p>'
        body += '<h3>任务歧义候选</h3><pre>'+html.escape(json.dumps(rubric['risks'],ensure_ascii=False,indent=2))+'</pre>'
        for r in sorted(grouped[tid],key=lambda x:(x['arm'],x['trial'])):
            title=f"{r['arm']} · trial {r['trial']} · reward {r['reward']} · "+'、'.join(CATEGORIES[c] for c in r['machine_categories'])
            body += '<details><summary>'+html.escape(title)+'</summary><p>以下为机器候选及独立证据，非官方重评分。</p><pre>'+html.escape(json.dumps({k:r[k] for k in ['evidence_override','outcome_tags','review','quarantined_findings','replay']},ensure_ascii=False,indent=2))+'</pre>'
            body += f'<a href="cases/{r["record_id"]}.md">完整消息及证据记录</a></details>'
        cards.append('<details class="task"><summary>'+html.escape(f"{tid}｜Base {counts['base']}/4 → SFT {counts['sft1']}/4｜"+rubric['goal'])+'</summary>'+body+'</details>')
    save(output/'task_checklists.json', task_machine)
    header = '# selection60 逐题检查表与失败分类\n\n'+GUIDE+'\n\n'
    header += '## 覆盖与校验\n\n```json\n'+json.dumps({k:summary[k] for k in ['tasks','failures','controls','normalized_cases','quarantined_cases','quarantined_findings','replay_status','saved_db_disagreements','scoring_checks_absent']},ensure_ascii=False,indent=2)+'\n```\n\n'
    header += '## 机器候选分类（可多标签，不代表确定根因）\n\n|分类|Base 失败109条|SFT 失败119条|\n|---|---|---|\n'
    for c,label in CATEGORIES.items():
        header += f"|{label}|{stats['base']['machine_candidate_categories'].get(c,0)}|{stats['sft1']['machine_candidate_categories'].get(c,0)}|\n"
    header += '\n## 独立证据复核案例\n\n'
    for o in overrides['cases']:
        r=next(r for r in records if (r['arm'],r['task_id'],r['trial'])==(o['arm'],o['task_id'],o['trial']))
        header += f"- [{o['task_id']} / {o['trial']} / {o['arm']}](cases/{r['record_id']}.md)："+'；'.join(o['notes'])+'\n'
    header += '\n## 60题索引\n\n|任务|Base|SFT 1 epoch|检查项|判读轨迹数（含抽样成功）|\n|---|---|---|---|---|\n'+'\n'.join(task_rows)+'\n'
    header += '\n## 工具任务覆盖（每组分母60题，实际调用不等于能力通过）\n\n|工具|gold参考|Base实际|SFT实际|\n|---|---|---|---|\n'
    for name, counts in sorted(tool_coverage.items()):
        header += f"|{name}|{counts['gold_task_count']}|{counts['base_task_count']}|{counts['sft1_task_count']}|\n"
    header += '\n## 成本与复现\n\n'+json.dumps(summary['api'],ensure_ascii=False)+'\n\n累计金额为保守用量估算，非账单；未触发新训练或模型推理评测。原始 calls、rubrics、reviews、冻结哈希及失败判读均保留；无付费重试。\n'
    (output/'README.md').write_text(header)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>selection60 逐题审计</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;background:#f7f8fa;color:#182434}details{background:white;border:1px solid #ccd5df;border-radius:8px;margin:12px 0;padding:14px}summary{cursor:pointer;font-weight:600}table{border-collapse:collapse;width:100%}td,th{border:1px solid #d7dee5;padding:8px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}input{padding:12px;width:85%;font-size:16px}aside{background:#fff4d6;padding:18px;border-radius:8px}</style><h1>selection60 逐题检查表与失败分类</h1><p>60题 · 228条失败全覆盖 · 12条成功盲审对照 · Base 131/240，SFT 1 epoch 121/240</p><aside>'+html.escape(GUIDE).replace('\n','<br>')+'</aside><p><a href="README.md">审计摘要</a> · <a href="failure_classification.jsonl">228条失败完整记录</a> · <a href="task_checklists.json">60题检查表 JSON</a></p><input id="filter" placeholder="搜索任务ID、支付、行李、评分问题…"><div id="cards">'+''.join(cards)+'</div><script>document.getElementById("filter").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll(".task").forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(q));});</script></html>'
    (output/'index.html').write_text(page)
    print(json.dumps({k:summary[k] for k in ['tasks','failures','normalized_cases','quarantined_cases','replay_status','saved_db_disagreements']},ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    render(args.root,args.output)

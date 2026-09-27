"""Read-only comparison of a frozen dual-metric model queue; never edits scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau3_grpo.evaluation.compare import compare_evaluations
from tau3_grpo.evaluation.dual_metrics import build_report, FIELDS
from tau3_grpo.evaluation.metrics import paired_bootstrap


def report(root, output):
    root,output=Path(root),Path(output)
    plan=json.loads((root/'plan.json').read_text())
    by_arm={};cases={};missing=[]
    for arm in plan['arms']:
        name=arm['name'];directory=Path(arm['output'])
        if not (directory/'run.json').exists():missing.append(name);continue
        by_arm[name],rows=build_report(directory)
        cases[name]={(r['task_id'],r['trial'],r['seed']):r for r in rows}
    names=[a['name'] for a in plan['arms']]
    comparisons={}
    for name in names:
        if name=='base':continue
        if name not in by_arm or 'base' not in by_arm:continue
        base=next(a for a in plan['arms'] if a['name']=='base')
        target=next(a for a in plan['arms'] if a['name']==name)
        protocol=compare_evaluations(Path(base['output']),Path(target['output']),ks=[1])
        item={'comparable':protocol['comparable'],'protocol':protocol['protocol'],'differences':None}
        if item['comparable']:
            item['differences']={}
            for field in FIELDS:
                b=by_arm['base']['metrics'][field]['per_task'];t=by_arm[name]['metrics'][field]['per_task']
                if set(b)!=set(t):raise ValueError('Different tasks; never compare an intersection')
                tasks=sorted(b)
                result=paired_bootstrap([b[x]['metrics']['pass@1'] for x in tasks],
                                        [t[x]['metrics']['pass@1'] for x in tasks],seed=42)
                item['differences'][field]=dict(result.to_dict(),difference_pp=100*result.difference,
                                                ci_pp=[100*result.ci_low,100*result.ci_high])
        comparisons[name]=item
    parent_comparisons={}
    if 'sft1' in by_arm and 'repair_sft1' in by_arm:
        left=next(a for a in plan['arms'] if a['name']=='sft1')
        right=next(a for a in plan['arms'] if a['name']=='repair_sft1')
        parent_comparisons['repair_sft1_vs_sft1']=compare_evaluations(Path(left['output']),Path(right['output']),ks=[1])
        parent=parent_comparisons['repair_sft1_vs_sft1']
        parent['dual_differences']=None
        if parent['comparable']:
            parent['dual_differences']={}
            for field in FIELDS:
                b=by_arm['sft1']['metrics'][field]['per_task'];t=by_arm['repair_sft1']['metrics'][field]['per_task']
                if set(b)!=set(t):raise ValueError('Different parent task sets')
                tasks=sorted(b)
                delta=paired_bootstrap([b[x]['metrics']['pass@1'] for x in tasks],
                                       [t[x]['metrics']['pass@1'] for x in tasks],seed=42)
                parent['dual_differences'][field]=dict(delta.to_dict(),difference_pp=100*delta.difference,
                                                       ci_pp=[100*delta.ci_low,100*delta.ci_high])
    result={'status':'complete' if not missing and all(r['metrics_valid'] for r in by_arm.values()) else 'incomplete',
            'missing_runs':missing,'runs':by_arm,'comparisons_to_base':comparisons,'comparisons_to_parent':parent_comparisons,
            'limits':['Single evaluation seed; task bootstrap does not cover training-seed variation.',
                      'Exposed repaired selection development smoke, not official unmodified final evaluation.',
                      'Semantic judge labels remain subject to manual review; no impressions substitute for success.']}
    output.mkdir(parents=True,exist_ok=False)
    (output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# '+' / '.join(names)+' 同轨迹双指标', '',f"状态：{result['status']}。",'',
           '|模型|计划数|已评分|严格成功|等价成功|严格率|等价率|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in plan['arms']:
        name=arm['name']
        if name not in by_arm:
            lines.append(f'|{name}|60|0|未运行|未运行|不可用|不可用|');continue
        s=[by_arm[name]['metrics'][f] for f in FIELDS]
        counts=[sum(t['successes'] for t in x['per_task'].values()) for x in s]
        rates=[f"{100*x['solve_rate']:.2f}%" if x['solve_rate'] is not None else '不可用' for x in s]
        lines.append(f"|{name}|{s[0]['planned_trajectories']}|{s[0]['completed_trajectories']}|{counts[0]}|{counts[1]}|{rates[0]}|{rates[1]}|")
    lines+=['','成功数来自已评分部分；存在未评分则完整集合成功率不可用。','',
            '## 逐题对照','', '|题目|'+'|'.join(n+' 严格/等价' for n in names)+'|', '|---|'+'---|'*len(names)]
    task_ids=[x['task_id'] for x in json.loads((Path(plan['bundle']).parent/'task_checklist.json').read_text())]
    for tid in task_ids:
        values=[]
        for name in names:
            c=cases.get(name,{}).get((tid,0,42))
            values.append(f"{c['reference_compliant']}/{c['equivalent_compliant']}" if c else '未运行')
        lines.append('|'+tid+'|'+'|'.join(values)+'|')
    lines+=['','## 配对差值（只有完整且协议一致才计算）','']
    for name,c in comparisons.items():
        if not c['comparable']:
            lines.append(f'- {name}：结果不完整或协议不一致，差值不可用。');continue
        for field,d in c['differences'].items():
            lines.append(f"- {name} − Base / {field}：{d['difference_pp']:+.2f} pp；任务配对95%区间 [{d['ci_pp'][0]:.2f}, {d['ci_pp'][1]:.2f}] pp。")
    lines+=['','## 从旧SFT1继续训练的变化','']
    for name,c in parent_comparisons.items():
        if not c['comparable']:
            lines.append(f'- {name}：结果不完整或协议不一致，差值不可用。')
        else:
            for field,d in c['dual_differences'].items():
                lines.append(f"- {name} / {field}：{d['difference_pp']:+.2f} pp；任务配对95%区间 [{d['ci_pp'][0]:.2f}, {d['ci_pp'][1]:.2f}] pp。")
    lines+=['','不覆盖训练随机性；多个指标和模型比较未作多重比较校正。',
            '终态集合外结果需查看工具证据，不能直接归因为训练差；分数不据人工印象修改。']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=report(a.root,a.output);print(json.dumps({'status':r['status'],'missing_runs':r['missing_runs']}))


if __name__=='__main__':main()

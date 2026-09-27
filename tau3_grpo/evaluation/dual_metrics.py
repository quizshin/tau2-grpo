"""Dual success rates and evidence-based triage of the same saved attempts.

No new rollout, paid judge, reference expansion, or reinterpretation of unknowns.
Failure labels are observations, not claims of verified causal attribution.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tau3_grpo.evaluation.scoring import summarize_trials

VERSION = 'airline_dual_success_v1'
FIELDS = ('reference_compliant', 'equivalent_compliant')


def receipt(row):
    data = (row.get('simulation', {}).get('info') or {}).get('outcome_contract')
    if not data or data.get('dual_metric_version') != VERSION:
        raise ValueError('Missing dual scoring receipt; cannot rename legacy reward')
    strict, equivalent = [data.get(k + '_reward') for k in FIELDS]
    if strict not in (0, 1) or equivalent not in (0, 1) or strict > equivalent or row['reward'] != equivalent:
        raise ValueError('Invalid dual metric nesting or primary reward')
    return data


def summarize_dual(*, planned, results, errors, trials, ks=None, include_pass_hat=False):
    reports = {}
    for field in FIELDS:
        rescored = [dict(row, reward=receipt(row)[field + '_reward']) for row in results]
        reports[field] = summarize_trials(planned=planned, results=rescored, errors=errors,
                                         trials=trials, ks=ks, include_pass_hat=include_pass_hat)
        reports[field]['success_rule'] = field + '; frozen airline_dual_success_v1'
    return {'version': VERSION, 'primary': FIELDS[1], 'metrics_valid': reports[FIELDS[1]]['metrics_valid'],
            'paired_same_trajectories': True, 'metrics': reports,
            'equivalent_only_successes': sum(receipt(r)[FIELDS[1] + '_reward'] > receipt(r)[FIELDS[0] + '_reward']
                                            for r in results)}


def case_record(row, *, scored):
    identity = {k: row[k] for k in ('task_id', 'trial', 'seed')}
    if not scored:
        message = row.get('error', '')
        kind = ('simulator_scope_unresolved' if 'Simulator scope' in message else
                'judge_unresolved' if row.get('error_type') == 'CommunicationUnresolved' or
                    (row.get('execution_evidence') or {}).get('stage') == 'communication_scoring' else
                'infrastructure_or_execution_unresolved')
        return dict(identity, scored=False, reference_compliant=None, equivalent_compliant=None,
                    labels=[kind], error_type=row.get('error_type'), error=message,
                    review_status='needs_manual_adjudication', source_file='errors.jsonl')
    data = receipt(row)
    behavior = data.get('behavior') or {}
    failed = [x for x in behavior.get('checks', []) if x.get('status') == 'fail']
    labels = []
    if data['reference_compliant_reward'] < data['equivalent_compliant_reward']:
        labels.append('accepted_equivalent_not_single_reference')
    if data.get('outcome_match') is False:
        labels.append('outside_frozen_outcome_set_requires_review')
    if data.get('action_policy_violations'):
        labels.append('deterministic_policy_violation')
    if any(x['id'].startswith('consent_') for x in failed):
        labels.append('judge_reported_consent_failure')
    if any(not x['id'].startswith('consent_') for x in failed):
        labels.append('judge_reported_goal_or_communication_failure')
    if behavior.get('structural_checks', {}).get('passed') is False:
        labels.append('required_tool_or_dependency_failure')
    if any(v < 1 - 1e-6 for v in data.get('non_db_components', {}).values()):
        labels.append('native_non_db_component_failure')
    from tau3_grpo.evaluation.eligibility import SCORABLE_TERMINATIONS
    if row.get('termination_reason') not in SCORABLE_TERMINATIONS:
        labels.append('execution_limit_or_agent_user_error')
    return dict(identity, scored=True, reference_compliant=data['reference_compliant_reward'],
                equivalent_compliant=data['equivalent_compliant_reward'], labels=labels,
                matched_variants=data.get('matched_variants', []),
                termination_reason=row.get('termination_reason'),
                policy_violations=data.get('action_policy_violations', []), failed_checks=failed,
                structural_checks=behavior.get('structural_checks'),
                legacy_reward_diagnostic_only=data.get('legacy_reward'),
                review_status='automated_evidence_not_manually_adjudicated', source_file='trajectories.jsonl')


def build_report(directory):
    from tau3_grpo.evaluation.artifacts import read_evaluation
    artifact = read_evaluation(Path(directory))
    spec = artifact.metadata['spec']
    summary = summarize_dual(planned=artifact.metadata['planned'], results=artifact.trajectories,
                             errors=artifact.errors, trials=spec['trials'], ks=spec.get('ks'),
                             include_pass_hat=spec.get('include_pass_hat', False))
    cases = [case_record(row, scored=True) for row in artifact.trajectories]
    cases += [case_record(row, scored=False) for row in artifact.errors]
    seen = {(x['task_id'], x['trial'], x['seed']) for x in cases}
    cases += [dict(job, scored=False, reference_compliant=None, equivalent_compliant=None,
                   labels=['missing_attempt'], review_status='unresolved')
              for job in artifact.metadata['planned'] if (job['task_id'],job['trial'],job['seed']) not in seen]
    cases.sort(key=lambda x: (x['task_id'], x['trial']))
    summary['diagnostic_label_counts'] = dict(Counter(label for x in cases for label in x['labels']))
    summary['artifact_sha256'] = artifact.files
    summary['limitations'] = [
        'Outcome mismatch alone is not proof of an agent mistake; an unenumerated legal result is possible.',
        'Language-model semantic findings need manual review; no score edits are made by this report.',
        'Smoke trials are not a stable improvement claim or official unmodified benchmark results.',
    ]
    return summary, cases


def write_report(directory, output):
    summary, cases = build_report(directory)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (output/'cases.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in cases))
    lines = ['# 双口径成功率与逐题检查', '', '同一轨迹、同一合规要求；仅终态接受集合不同。',
             '未评分不当作失败，也不从计划分母删除；任一未评分时不发布汇总成功率。', '',
             '|口径|成功数（已评分部分）|计划尝试|完整有效|成功率|','|---|---:|---:|---|---:|']
    for field in FIELDS:
        s=summary['metrics'][field]; successes=sum(t['successes'] for t in s['per_task'].values())
        value=f"{100*s['solve_rate']:.2f}%" if s['solve_rate'] is not None else '不可用'
        lines.append(f"|{field}|{successes}|{s['planned_trajectories']}|{s['metrics_valid']}|{value}|")
    lines += ['', '## 逐题检查（标签是证据提示，不是最终根因判决）', '',
              '|题目|trial|严格|等价|标签|', '|---|---:|---:|---:|---|']
    for c in cases:
        lines.append(f"|{c['task_id']}|{c['trial']}|{c['reference_compliant']}|{c['equivalent_compliant']}|{', '.join(c['labels']) or '两口径通过'}|")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(write_report(args.evaluation,args.output),ensure_ascii=False))


if __name__ == '__main__':
    main()

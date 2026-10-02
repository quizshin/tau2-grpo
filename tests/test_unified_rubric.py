"""Offline contracts: source identity, denominators, safe judge requests and reports."""
import asyncio
import copy
import hashlib
import json

import pytest

from tau3_grpo.evaluation.rubric import (
    judge,
    main,
    prepare,
    read_reviews,
    validate_package,
)
from tau3_grpo.evaluation.rubric_contract import (
    DEFINITIONS,
    DIMENSIONS,
    PROMPT,
    VERSION,
    bind_bundle,
    freeze_bundle,
    load_bundle,
    validate_review,
    visible_events,
    write_json,
)
from tau3_grpo.evaluation.rubric_report import build_report, compare_reports, save_report
from tau3_grpo.utils.hashing import sha256_json


@pytest.fixture
def bundle():
    policy = 'Frozen policy; confirm before mutating. Baggage allowance is per passenger.'
    schemas = [{'type': 'function', 'function': {'name': 'get_reservation_details'}}]
    return freeze_bundle({
        'version': VERSION, 'policy': policy,
        'agent_system_prompt_sha256': hashlib.sha256(policy.encode()).hexdigest(),
        'tool_schemas': schemas, 'tool_schemas_sha256': sha256_json(schemas),
        'tasks': {t: {'reviewed_by': 'test_fixture_only', 'criteria': dict(DEFINITIONS),
                      'capabilities': ['state_reading', 'argument_binding'],
                      'bucket': 'constraints', 'task_definition_sha256': 'a' * 64,
                      'task_db_sha256': {'representation': 'source_file_bytes', 'sha256': 'b' * 64}}
                  for t in ('a', 'b')},
    })


def run_fixture(path, bundle, *, missing=False, retrospective=False):
    plan = [{'task_id': t, 'trial': i, 'seed': 42 + i} for t in ('a', 'b') for i in range(2)]
    prov = {
        'agent_system_prompt_sha256': bundle['agent_system_prompt_sha256'],
        'rubric_tool_schemas_sha256': bundle['tool_schemas_sha256'],
        'task_definition_sha256': {t: v['task_definition_sha256'] for t, v in bundle['tasks'].items()},
        'task_db_sha256': {t: v['task_db_sha256'] for t, v in bundle['tasks'].items()},
        'task_manifest_sha256': 'manifest', 'evaluator_source_sha256': {'source': 'same'},
        'harness_source_sha256': {'source': 'same'}, 'checkpoint_hash': path.name,
        'policy_service_attestation_hash': path.name, 'harness_protocol': {'version': 'test'},
        'tool_protocol': 'sequential', 'benchmark_policy_modified': True,
    }
    if not retrospective:
        prov['rubric_bundle_sha256'] = bundle['bundle_sha256']
    meta = {'schema_version': 1, 'benchmark_revision': 'pinned', 'planned': plan,
            'spec': {'target': 'selection', 'trials': 2, 'ks': [1, 2], 'include_pass_hat': False,
                     'seed': 42, 'max_steps': 30, 'max_errors': 10, 'max_concurrency': 2},
            'provenance': prov,
            'endpoints': {'policy': {'model': path.name, 'temperature': .7, 'repetition_penalty': 1},
                          'user': {'model': 'user', 'temperature': .7}}}
    rows = [{**job, 'reward': 1, 'termination_reason': 'user_stop',
             'simulation': {'reward_info': {'reward': 1}, 'messages': [
                 {'role': 'user', 'content': 'Check reservation.'},
                 {'role': 'assistant', 'content': 'Confirmed.', 'reasoning_content': 'PRIVATE'},
             ]}} for job in plan]
    write_json(path / 'run.json', meta)
    (path / 'trajectories.jsonl').write_text('\n'.join(json.dumps(r) for r in (rows[:-1] if missing else rows)))
    return path


def answer(status='satisfied'):
    return {'dimensions': {d: {'status': status, 'reason': 'fixture judgment', 'evidence': ['m0001']}
                           for d in DIMENSIONS}}


class FakeJudge:
    provenance = {'provider': 'fixture', 'model': 'test-only', 'temperature': 0}

    def __init__(self, status='satisfied', error=False):
        self.calls = 0
        self.metadata = []
        self.status, self.error = status, error

    async def extract_json(self, request, *, user_payload):
        self.calls += 1
        assert request['system'] == PROMPT
        assert 'reward' not in json.dumps(user_payload)
        assert 'PRIVATE' not in json.dumps(user_payload)
        if self.error:
            raise RuntimeError('SECRET provider body')
        self.metadata.append({'total_tokens': 3})
        return answer(self.status)


def reviewed(run, bundle, output, *, status='satisfied', retrospective=False):
    package = prepare(run, bundle, retrospective=retrospective)
    asyncio.run(judge(package, output, client=FakeJudge(status), max_calls=4))
    return package


def test_unreviewed_draft_cannot_freeze(bundle):
    bundle['tasks']['a']['reviewed_by'] = ''
    with pytest.raises(ValueError, match='reviewer'):
        freeze_bundle(bundle)


@pytest.mark.parametrize('change', ['policy', 'criteria', 'db', 'schemas'])
def test_frozen_changes_rejected(tmp_path, bundle, change):
    if change == 'policy':
        bundle['policy'] += 'changed'
    elif change == 'criteria':
        bundle['tasks']['a']['criteria']['scope'] += 'changed'
    elif change == 'db':
        bundle['tasks']['a']['task_db_sha256']['sha256'] = 'c' * 64
    else:
        bundle['tool_schemas'].append({'other': 1})
    write_json(tmp_path / 'bundle.json', bundle)
    with pytest.raises(ValueError):
        load_bundle(tmp_path / 'bundle.json')


@pytest.mark.parametrize('field', ['task_definition_sha256', 'task_db_sha256', 'agent_system_prompt_sha256', 'rubric_tool_schemas_sha256'])
def test_wrong_runtime_inputs_rejected(tmp_path, bundle, field):
    run = run_fixture(tmp_path / 'run', bundle)
    meta = json.loads((run / 'run.json').read_text())
    meta['provenance'][field] = {} if field.startswith('task_') else 'different'
    (run / 'run.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='differ'):
        prepare(run, bundle)


def test_extra_task_rejected(tmp_path, bundle):
    run = run_fixture(tmp_path / 'run', bundle)
    meta = json.loads((run / 'run.json').read_text())
    with pytest.raises(ValueError, match='task set'):
        bind_bundle(bundle, meta['provenance'], ['a'])


def test_visible_events_exclude_nested_metadata_and_keep_correlation():
    messages = [{'role': 'system', 'content': 'evil'}, {'role': 'assistant', 'content': None,
                 'tool_calls': [{'id': 'x', 'reasoning': 'PRIVATE', 'function': {
                     'name': 'read', 'arguments': {'id': 'r'}, 'reward': 42}}]},
                {'role': 'tool', 'content': 'not found', 'id': 'x', 'error': True,
                 'requestor': 'assistant', 'metadata': {'gold': 1}}]
    events = visible_events(messages)
    assert [r['event_id'] for r in events] == ['m0001', 'm0002']
    assert events[-1]['id'] == 'x' and events[-1]['error']
    assert not any(word in json.dumps(events) for word in ('PRIVATE', 'reward', 'gold', 'evil'))
    events[0]['tool_calls'][0]['arguments']['id'] = 'changed'
    assert messages[1]['tool_calls'][0]['function']['arguments']['id'] == 'r'


@pytest.mark.parametrize('defect', ['missing_dimension', 'old_score', 'foreign_evidence', 'empty_evidence', 'no_reason'])
def test_invalid_judge_verdict_rejected(defect):
    value = answer()
    if defect == 'missing_dimension':
        value['dimensions'].pop('scope')
    elif defect == 'old_score':
        value['dimensions']['scope']['status'] = 2
    elif defect == 'foreign_evidence':
        value['dimensions']['scope']['evidence'] = ['future_event']
    elif defect == 'empty_evidence':
        value['dimensions']['scope']['evidence'] = []
    else:
        value['dimensions']['scope']['reason'] = ''
    with pytest.raises(ValueError):
        validate_review(value, {'events': [{'event_id': 'm0001'}]})


def test_complete_chain_success_can_still_be_badcase(tmp_path, bundle):
    run = run_fixture(tmp_path / 'run', bundle)
    reviews = tmp_path / 'reviews'
    reviewed(run, bundle, reviews, status='violated')
    report = build_report(run, bundle, reviews)
    assert report['outcome']['metrics_valid']
    assert report['rubric_coverage_complete'] and report['pre_run_bound']
    assert len(report['badcases']) == 4
    assert all(c['outcome_success'] and 'rubric_violation_candidate' in c['flags'] for c in report['badcases'])
    assert report['capability_slices']['state_reading']['planned_trials'] == 4
    assert report['task_macro']['policy']['rate'] == 0
    save_report(tmp_path / 'report', report)
    assert len((tmp_path / 'report/badcases.jsonl').read_text().splitlines()) == 4


@pytest.mark.parametrize('status', ['unknown', 'not_applicable'])
def test_unknown_and_na_are_not_pass_or_fail(tmp_path, bundle, status):
    run = run_fixture(tmp_path / 'run', bundle)
    reviews = tmp_path / 'reviews'
    reviewed(run, bundle, reviews, status=status)
    report = build_report(run, bundle, reviews)
    counts = report['overall']['dimensions']['scope']
    assert counts[status] == 4
    assert counts['satisfied'] == counts['violated'] == 0
    assert counts['known_verdict_satisfaction_rate'] is None
    assert report['task_macro']['scope']['rate'] is None


def test_missing_trial_never_dropped(tmp_path, bundle):
    run = run_fixture(tmp_path / 'run', bundle, missing=True)
    reviews = tmp_path / 'reviews'
    package = reviewed(run, bundle, reviews)
    assert len(package['requests']) == 3 and len(package['unavailable']) == 1
    report = build_report(run, bundle, reviews)
    assert not report['outcome']['metrics_valid'] and not report['rubric_coverage_complete']
    assert report['overall']['planned_trials'] == 4
    assert report['overall']['dimensions']['scope']['unjudged'] == 1
    assert report['task_macro']['scope']['rate'] is None


def test_missing_judge_response_preserves_outcome(tmp_path, bundle):
    run = run_fixture(tmp_path / 'run', bundle)
    reviews = tmp_path / 'reviews'
    reviewed(run, bundle, reviews)
    next(reviews.glob('*.response.json')).unlink()
    report = build_report(run, bundle, reviews)
    assert report['outcome']['metrics_valid'] and not report['rubric_coverage_complete']
    assert report['overall']['dimensions']['completion']['unjudged'] == 1


def test_duplicate_and_foreign_response_rejected(tmp_path, bundle):
    run = run_fixture(tmp_path / 'run', bundle)
    reviews = tmp_path / 'reviews'
    package = reviewed(run, bundle, reviews)
    row = json.loads(next(reviews.glob('*.response.json')).read_text())
    write_json(reviews / 'duplicate.response.json', row)
    with pytest.raises(ValueError, match='duplicate'):
        read_reviews(reviews, package)
    row['request_sha256'] = 'foreign'
    (reviews / 'duplicate.response.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Unknown'):
        read_reviews(reviews, package)


def test_judge_budget_no_retry_and_durable_failure(tmp_path, bundle):
    run = run_fixture(tmp_path / 'run', bundle)
    package = prepare(run, bundle)
    fake = FakeJudge(error=True)
    with pytest.raises(ValueError, match='max_calls'):
        asyncio.run(judge(package, tmp_path / 'reviews', client=fake, max_calls=3))
    assert fake.calls == 0 and not (tmp_path / 'reviews').exists()
    asyncio.run(judge(package, tmp_path / 'reviews', client=fake, max_calls=4))
    assert fake.calls == 4
    assert len(list((tmp_path / 'reviews').glob('*.pending.json'))) == 4
    assert all('SECRET' not in p.read_text() for p in (tmp_path / 'reviews').glob('*.json'))
    with pytest.raises(FileExistsError):
        asyncio.run(judge(package, tmp_path / 'reviews', client=fake, max_calls=4))
    assert fake.calls == 4


def test_mutated_requests_rejected_before_calls(tmp_path, bundle):
    package = prepare(run_fixture(tmp_path / 'run', bundle), bundle)
    package['requests'][0]['criteria']['scope'] = 'always pass'
    with pytest.raises(ValueError, match='changed'):
        validate_package(package)


def test_run_changed_after_judging_invalidates_reviews(tmp_path, bundle):
    run = run_fixture(tmp_path / 'run', bundle)
    reviews = tmp_path / 'reviews'
    reviewed(run, bundle, reviews)
    with (run / 'trajectories.jsonl').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='different requests'):
        build_report(run, bundle, reviews)


def test_retrospective_requires_explicit_flag(tmp_path, bundle):
    run = run_fixture(tmp_path / 'run', bundle, retrospective=True)
    with pytest.raises(ValueError, match='retrospective'):
        prepare(run, bundle)
    assert not prepare(run, bundle, retrospective=True)['pre_run_bound']


def test_compare_requires_same_judge_and_decisive_denominators(tmp_path, bundle):
    left, right = (run_fixture(tmp_path / p, bundle) for p in ('left', 'right'))
    lr, rr = tmp_path / 'lr', tmp_path / 'rr'
    reviewed(left, bundle, lr)
    reviewed(right, bundle, rr, status='violated')
    report = compare_reports(left, right, bundle, lr, rr)
    assert report['comparable']
    assert report['diagnostic_task_macro_delta']['policy'] == -1
    manifest = json.loads((rr / 'manifest.json').read_text())
    manifest['judge']['model'] = 'another'
    (rr / 'manifest.json').write_text(json.dumps(manifest))
    assert not compare_reports(left, right, bundle, lr, rr)['comparable']


def test_offline_cli_freeze_prepare_report(tmp_path, bundle):
    run = run_fixture(tmp_path / 'run', bundle)
    draft = copy.deepcopy(bundle)
    draft.pop('bundle_sha256')
    write_json(tmp_path / 'draft.json', draft)
    assert main(['freeze', '--draft', str(tmp_path / 'draft.json'), '--output', str(tmp_path / 'sealed.json')]) == 0
    assert main(['prepare', '--run-dir', str(run), '--bundle', str(tmp_path / 'sealed.json'),
                 '--output', str(tmp_path / 'requests.json')]) == 0
    reviewed(run, bundle, tmp_path / 'reviews')
    assert main(['report', '--run-dir', str(run), '--bundle', str(tmp_path / 'sealed.json'),
                 '--reviews', str(tmp_path / 'reviews'), '--output', str(tmp_path / 'report')]) == 0

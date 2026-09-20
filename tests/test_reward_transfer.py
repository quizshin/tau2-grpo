import json
import subprocess
import sys
from copy import deepcopy

import pytest

from tau3_grpo.analysis.audit_reward_transfer import (
    adapt_row,
    describe,
    feature_row,
    main,
    simulation_turns,
)
from tau3_grpo.analysis.calibrate_paper_rewards import implementation_hashes
from tau3_grpo.evaluation.process_reward import reward_settings, score_turns

RECIPE = reward_settings({'mode': 'paper', 'version': 'paper_env_v2'})
CALL = {'id': 'a', 'name': 'get_user_details', 'arguments': {'user_id': 'U'}, 'requestor': 'assistant'}
ENTRY = {'task_id': 't', 'split': 'selection', 'task': {'evaluation_criteria': {'actions': [CALL]}}}


def sample():
    return {'task_id': 't', 'seed': 42, 'trial': 0, 'reward': 1, 'termination_reason': 'user_stop',
            'simulation': {'task_id': 't', 'seed': 42, 'trial': None, 'termination_reason': 'user_stop',
                           'reward_info': {'reward': 1, 'reward_basis': ['DB']},
                           'messages': [
                               {'role': 'user', 'content': 'hello'},
                               {'role': 'assistant', 'tool_calls': [deepcopy(CALL)]},
                               {'role': 'tool', 'id': 'a', 'error': False, 'content': '{"x":1}'},
                               {'role': 'assistant', 'content': 'done'},
                           ]}}


def test_multicall_rounds_and_responses_are_preserved():
    row = sample()
    m = row['simulation']['messages']
    m[1]['tool_calls'].append({**CALL, 'id': 'b', 'arguments': {'user_id': 'V'}})
    m.insert(3, {'role': 'tool', 'id': 'b', 'error': True, 'content': 'user missing'})
    turns = simulation_turns(row['simulation'])
    assert len(turns) == 2
    assert [e['id'] for e in turns[0]['tool_calls']] == ['a', 'b']
    assert turns[0]['tool_calls'][1]['error'] is True
    process, outcome, scored, kind = adapt_row(row, ENTRY, RECIPE)
    assert process['turn_records'][0]['reward_types'] == ['gold_exact', 'error']
    assert outcome == 1 and scored
    assert 'synthetic' in kind
    assert 'reward_type' not in m[1]['tool_calls'][0]  # original evidence unchanged


@pytest.mark.parametrize('change', ['missing_response', 'wrong_id', 'duplicate_id', 'missing_error',
                                    'missing_content', 'orphan', 'early_user', 'early_assistant'])
def test_missing_or_ambiguous_execution_evidence_is_rejected(change):
    s = sample()['simulation']
    m = s['messages']
    if change == 'missing_response':
        m.pop(2)
    elif change == 'wrong_id':
        m[2]['id'] = 'wrong'
    elif change == 'duplicate_id':
        m[1]['tool_calls'].append(deepcopy(CALL))
    elif change == 'missing_error':
        del m[2]['error']
    elif change == 'missing_content':
        m[2]['content'] = None
    elif change == 'orphan':
        m.insert(3, deepcopy(m[2]))
    elif change == 'early_user':
        m.insert(2, {'role': 'user', 'content': 'hi'})
    else:
        m.insert(2, {'role': 'assistant', 'content': 'hi'})
    with pytest.raises(ValueError):
        simulation_turns(s)


def test_changed_repeated_read_and_error_retry_use_original_observations():
    row = sample()
    m = row['simulation']['messages']
    m[2]['error'] = True
    m[3:] = [
        {'role': 'assistant', 'tool_calls': [{**CALL, 'id': 'b'}]},
        {'role': 'tool', 'id': 'b', 'error': False, 'content': '{"x":1}'},
        {'role': 'assistant', 'tool_calls': [{**CALL, 'id': 'c'}]},
        {'role': 'tool', 'id': 'c', 'error': False, 'content': '{"x":2}'},
    ]
    process, *_ = adapt_row(row, ENTRY, RECIPE)
    assert [t['reward_types'] for t in process['turn_records']] == [['error'], ['gold_exact'], ['read_only']]
    m[-1]['observation_truncated'] = True
    process, *_ = adapt_row(row, ENTRY, RECIPE)
    assert process['turn_records'][-1]['reward_types'] == ['unknown']


def test_nonsequential_response_order_is_not_silently_reordered():
    row = sample()
    m = row['simulation']['messages']
    m[1]['tool_calls'].append({**CALL, 'id': 'b'})
    m.insert(2, {'role': 'tool', 'id': 'b', 'error': False, 'content': 'ok'})
    with pytest.raises(ValueError, match='order'):
        simulation_turns(row['simulation'])


def test_logged_replay_preserves_unscored_status_and_detects_tampering():
    p = score_turns(simulation_turns(sample()['simulation']), [CALL], ['DB'], RECIPE, official_outcome=0)
    row = {'task_id': 't', 'score': 0, 'scored': False, 'process_reward_json': json.dumps(p)}
    _, outcome, scored, kind = adapt_row(row, ENTRY, RECIPE)
    assert outcome == 0 and not scored and kind == 'logged_process_replayed'
    p['turn_rewards'][0] = 99
    row['process_reward_json'] = json.dumps(p)
    with pytest.raises(ValueError, match='replay'):
        adapt_row(row, ENTRY, RECIPE)


def test_text_only_rows_are_not_fabricated_and_labels_are_checked():
    with pytest.raises(ValueError, match='text-only'):
        adapt_row({'task_id': 't', 'score': 0, 'output': 'tool error'}, ENTRY, RECIPE)
    row = sample()
    row['simulation']['reward_info']['reward'] = 0
    with pytest.raises(ValueError, match='reward'):
        adapt_row(row, ENTRY, RECIPE)


def test_premature_simulation_null_basis_keeps_recorded_zero_but_is_not_officially_scored():
    row = sample()
    row.update(reward=0, termination_reason='max_steps')
    row['simulation']['termination_reason'] = 'max_steps'
    row['simulation']['reward_info'].update(reward=0, reward_basis=None)
    process, outcome, scored, _ = adapt_row(row, ENTRY, RECIPE)
    assert outcome == 0 and not scored
    assert process['reward_basis'] == []
    row['termination_reason'] = row['simulation']['termination_reason'] = 'user_stop'
    with pytest.raises(ValueError, match='basis'):
        adapt_row(row, ENTRY, RECIPE)


@pytest.mark.parametrize('aggregation', ['sum', 'mean'])
def test_component_decomposition_and_unscored_sensitivity(aggregation):
    config = deepcopy(RECIPE)
    config['paper_options']['aggregation'] = aggregation
    row = sample()
    m = row['simulation']['messages']
    m[1]['tool_calls'].append({**CALL, 'id': 'b', 'name': 'cancel_reservation',
                              'arguments': {'reservation_id': 'R'}})
    m.insert(3, {'role': 'tool', 'id': 'b', 'error': False, 'content': 'done'})
    process, *_ = adapt_row(row, ENTRY, config)
    f = feature_row(process, task_id='t', outcome=1, scored=True)
    assert f['mean_reward'] == pytest.approx(f['gold_read_mean'] + f['state_change_mean'])
    assert f['gold_read_calls'] == 1
    summary = describe([f, {**f, 'outcome': 0, 'scored': False}, {**f, 'outcome': 0, 'scored': True}])
    assert summary['recorded_outcomes']['rows'] == 3
    assert summary['officially_scored_only']['rows'] == 2
    stat = summary['recorded_outcomes']['features']['state_change_mean']
    assert stat['negative'] == 3 and stat['positive'] == stat['zero'] == 0


def test_cli_fixed_recipe_audit_and_final_manifest_rejection(tmp_path):
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps({'implementation_sha256': implementation_hashes(),
                                  'rounds': [{'candidate_recipe': RECIPE}]}))
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps(ENTRY))
    buffer = tmp_path / 'rows.jsonl'
    buffer.write_text(json.dumps(sample()))
    args = ['--recipe-report', str(recipe), '--manifest', str(manifest), '--split', 'selection',
            '--input', str(buffer)]
    output = tmp_path / 'audit'
    assert main([*args, '--output-dir', str(output)]) == 0
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'diagnostic_only'
    assert report['recipe'] == RECIPE
    assert not (output / 'frozen-recipe.json').exists()
    manifest.write_text(json.dumps({**ENTRY, 'split': 'final'}))
    with pytest.raises(ValueError, match='split mismatch'):
        main([*args, '--output-dir', str(tmp_path / 'final')])


def test_offline_cli_imports_without_test_suite_environment_initialization():
    result = subprocess.run([sys.executable, '-m', 'tau3_grpo.analysis.audit_reward_transfer', '--help'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr

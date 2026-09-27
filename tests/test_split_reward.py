from copy import deepcopy

import numpy as np
import pytest
import yaml

from tau3_grpo.analysis.calibrate_paper_rewards import (
    calibrate_round,
    validate_irc,
)
from tau3_grpo.evaluation.process_reward import DEFAULT_WEIGHTS, reward_settings, score_turns
from tau3_grpo.paths import CODE_ROOT

VERSION = 'paper_env_split_v3'
READ = {'name': 'get_reservation_details', 'arguments': {'reservation_id': 'A'}}
WRITE = {'name': 'cancel_reservation', 'arguments': {'reservation_id': 'A'}}


def score(calls, gold, version=VERSION, outcome=0):
    turns = [{'schema': 'tau3_turn_v1', 'turn_index': k, 'token_span': [k, k + 1],
              'tool_calls': deepcopy(events)} for k, events in enumerate(calls)]
    return score_turns(turns, gold, ['DB'], {'mode': 'paper', 'version': version}, official_outcome=outcome)


def event(call, **kwargs):
    return {**deepcopy(call), 'error': False, 'observation': '{"state":"initial"}', **kwargs}


def test_split_exact_reads_writes_errors_and_duplicates_without_legacy_changes():
    calls = [[event(READ)], [event(WRITE, error=True)], [event(WRITE)],
             [event(READ, observation='{"state":"cancelled"}')], [event(WRITE)]]
    gold = [READ, WRITE]
    split = score(calls, gold)
    assert [t['reward_types'] for t in split['turn_records']] == [
        ['gold_read'], ['error'], ['gold_write'], ['read_only'], ['duplicate']]
    assert split['turn_rewards'] == [0, -.1, 1, 0, 0]
    assert score_turns(split['turn_records'], gold, ['DB'], split['settings'], official_outcome=0) == split
    legacy = score(calls, gold, 'paper_env_v2')
    assert [t['reward_types'] for t in legacy['turn_records']] == [
        ['gold_exact'], ['error'], ['gold_exact'], ['read_only'], ['duplicate']]
    assert legacy['turn_rewards'] == [1, -.1, 1, 0, -.2]
    assert legacy['settings']['weights'] == DEFAULT_WEIGHTS
    assert 'reward_type' not in calls[0][0]


def test_non_db_reference_action_is_not_gold_write_and_split_schema_is_explicit():
    handoff = {'name': 'transfer_to_human_agents', 'arguments': {'summary': 'help'}}
    result = score([[event(handoff)]], [handoff])
    assert result['turn_records'][0]['reward_types'] == ['gold_exact']
    assert result['turn_rewards'] == [0]
    with pytest.raises(ValueError, match='tier'):
        reward_settings({'mode': 'paper', 'version': 'paper_env_v2', 'weights': {'gold_write': 1}})


def test_split_fit_requires_write_support_and_does_not_fit_saturated_read():
    cfg = yaml.safe_load((CODE_ROOT / 'configs/analysis/mt_gtpo_irc_paper_split_20260918.yaml').read_text())
    validate_irc(cfg)
    cfg['min_support'] = 2
    # All trajectories have reference reads; only successes match reference writes.
    cfg['intended_signs'] = {'gold_write': 1}
    cfg['fixed_weights'].update(error=0., state_change=0.)
    processes, outcomes, tasks = [], [], []
    for task in 'abcd':
        for success in [1, 0, 1, 0]:
            calls = [[event(READ)], [event(WRITE if success else {**WRITE, 'arguments': {'reservation_id': 'B'}})]]
            processes.append(score(calls, [READ, WRITE], outcome=success))
            outcomes.append(success)
            tasks.append(task)
    update = {'processes': processes, 'outcomes': outcomes, 'task_ids': tasks, 'uids': tasks,
              'algorithm': {'gamma': .9, 'lambda_outcome': .3, 'eps': 1e-6, 'min_group_size': 2}}
    recipe = reward_settings({'mode': 'paper', 'version': VERSION})
    result = calibrate_round([update], recipe, {'a', 'b'}, {'c', 'd'}, cfg)
    assert result['passed']
    assert result['candidate_recipe']['weights']['gold_read'] == 0
    assert result['candidate_recipe']['weights']['gold_write'] == pytest.approx(1)
    assert not result['initial_calibration']['tiers']['gold_read']['supported']
    changed = deepcopy(update)
    changed['outcomes'][8:] = [1 - x for x in changed['outcomes'][8:]]
    assert calibrate_round([changed], recipe, {'a', 'b'}, {'c', 'd'}, cfg)['candidate_recipe'] == result['candidate_recipe']
    cfg['min_support'] = 20
    assert not calibrate_round([update], recipe, {'a', 'b'}, {'c', 'd'}, cfg)['passed']
    del cfg['reward_version']
    with pytest.raises(ValueError, match='tier'):
        calibrate_round([update], recipe, {'a', 'b'}, {'c', 'd'}, cfg)


@pytest.mark.parametrize('version', [VERSION, 'paper_env_split_v4'])
@pytest.mark.parametrize('enabled', [False, True])
def test_split_verl_roundtrip_and_filter_keep_process_only_signal(enabled, version):
    import json

    import torch

    from tau3_grpo.algorithms.mt_gtpo_verl import compute_mt_gtpo_verl, last_stats
    from tau3_grpo.analysis.calibrate_process_rewards import analyze_update, summarize

    processes = [score([[event(READ), event(WRITE if i == 0 else READ)]], [READ, WRITE], version) for i in range(4)]
    metadata = {'uid': np.array(['a', 'a', 'b', 'b']),
                'process_reward_json': np.array([json.dumps(p) for p in processes], dtype=object)}
    mask = torch.ones((4, 1))
    config = {'process_reward': processes[0]['settings'],
              'dynamic_filter': {'enable': enabled, 'group_size': 2}}
    adv, _ = compute_mt_gtpo_verl(torch.zeros((4, 1)), mask, metadata['uid'], config, metadata)
    assert adv[0, 0] > 0 and adv[1, 0] < 0
    assert mask.sum() == (2 if enabled else 4)
    assert last_stats()['candidate_calls/gold_read'] == 4
    assert last_stats()['candidate_calls/gold_write'] == 1
    assert last_stats()['candidate_call_reward_sum/gold_read'] == 0
    assert last_stats()['candidate_call_reward_sum/gold_write'] == 1
    replay = [{'mt_gtpo_replay_json': value} for value in metadata['mt_gtpo_replay_json']]
    assert summarize(analyze_update(replay))['tiers']['gold_write']['trajectory_count'] == 1
    bad = json.loads(metadata['process_reward_json'][0])
    bad['official_outcome'] = 1
    metadata['process_reward_json'][0] = json.dumps(bad)
    with pytest.raises(ValueError, match='outcome mismatch'):
        compute_mt_gtpo_verl(torch.zeros((4, 1)), mask, metadata['uid'], config, metadata)


@pytest.mark.parametrize('summary', ['help', 'different wording'])
def test_v4_handoff_neutral_exact_unmatched_repeated_and_error(summary):
    gold = {'name': 'transfer_to_human_agents', 'arguments': {'summary': 'help'}}
    actual = {**gold, 'arguments': {'summary': summary}}
    calls = [[event(actual, error=True)], [event(actual)], [event(actual)],
             [event(WRITE)], [event({**WRITE, 'arguments': {'reservation_id': 'B'}})]]
    result = score(calls, [gold, WRITE], 'paper_env_split_v4')
    assert [t['reward_types'] for t in result['turn_records']] == [
        ['error'], ['generic'], ['generic'], ['gold_write'], ['state_change']]
    assert result['turn_rewards'] == [-.1, 0, 0, 1, -.1]
    assert score_turns(result['turn_records'], result['golden_actions'], ['DB'],
                       result['settings'], official_outcome=0) == result
    legacy = score(calls, [gold, WRITE])
    assert legacy['turn_records'][1]['reward_types'] == (['gold_exact'] if summary == 'help' else ['state_change'])
    assert legacy['turn_records'][2]['reward_types'] == ['duplicate']
    assert all('reward_type' not in e for turn in calls for e in turn)


def test_v4_generic_weight_cannot_be_fitted_or_overridden():
    with pytest.raises(ValueError, match='neutral'):
        reward_settings({'mode': 'paper', 'version': 'paper_env_split_v4', 'weights': {'generic': .1}})
    with pytest.raises(ValueError, match='tier'):
        reward_settings({'mode': 'paper', 'version': VERSION, 'weights': {'generic': 0}})
    config = yaml.safe_load((CODE_ROOT / 'configs/analysis/mt_gtpo_irc_paper_split_v4.yaml').read_text())
    validate_irc(config)
    assert config['fixed_weights']['generic'] == 0
    del config['fixed_weights']['generic']
    config['intended_signs']['generic'] = -1
    with pytest.raises(ValueError, match='fixed at zero'):
        validate_irc(config)


def test_v4_unknown_tool_remains_unknown_and_calculate_remains_read():
    calls = [[event({'name': 'unknown_tool', 'arguments': {}})],
             [event({'name': 'calculate', 'arguments': {'expression': '1+1'}})]]
    assert [t['reward_types'] for t in score(calls, [], 'paper_env_split_v4')['turn_records']] == [
        ['unknown'], ['read_only']]

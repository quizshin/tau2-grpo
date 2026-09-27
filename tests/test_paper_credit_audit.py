import json

import numpy as np
import pytest

from tau3_grpo.analysis.audit_paper_credit import (
    argument_differences,
    association,
    decompose,
    duplicate_history,
    execution_comparison,
    main,
    official_tool_types,
)

SETTINGS = {"gamma": .9, "lambda_outcome": .3, "eps": 1e-6, "min_group_size": 2}


def process(turns, aggregation="sum"):
    return {"settings": {"paper_options": {"aggregation": aggregation}},
            "turn_records": [{"tool_calls": [{"reward_type": tier, "reward": value}
                                             for tier, value in turn]} for turn in turns],
            "turn_rewards": [sum(v for _, v in t) / (len(t) if aggregation == "mean" else 1)
                             for t in turns]}


@pytest.mark.parametrize("aggregation", ["sum", "mean"])
def test_exact_decomposition_future_can_outweigh_negative_duplicate(aggregation):
    processes = [process([[('duplicate', -.2), ('read_only', 0)], [('gold_exact', 1)]], aggregation),
                 process([[('read_only', 0)], [('read_only', 0)]], aggregation)]
    rows, error = decompose(processes, [1, 0], ['g', 'g'], SETTINGS)
    row = rows[0]
    assert row['reward'] < 0 < row['advantage']
    assert row['components']['duplicate_immediate'] < 0
    assert row['components']['future_process'] > 0
    assert row['components']['terminal'] > 0
    assert row['episode_component'] > 0
    expected_return = processes[0]['turn_rewards'][0] + .9 + .9**2
    std = np.std([expected_return, 0])
    assert row['components']['duplicate_immediate'] == pytest.approx(
        (processes[0]['turn_rewards'][0] / 2) / (std + SETTINGS['eps']))
    assert error < 1e-12


def test_co_turn_gold_and_worse_peers_can_produce_positive_credit():
    rows, _ = decompose([process([[('duplicate', -.2), ('gold_exact', 1)]]),
                         process([[('error', -.5)]])], [0, 0], ['g', 'g'], SETTINGS)
    assert rows[0]['advantage'] > 0
    assert rows[0]['components']['duplicate_immediate'] < 0
    assert rows[0]['components']['other_immediate'] > 0
    assert rows[0]['components']['future_process'] == 0
    assert rows[0]['components']['terminal'] == 0


def test_zero_variance_singletons_and_missing_turns():
    processes = [process([[('duplicate', -.2)], [('gold_exact', 1)]]),
                 process([[('duplicate', -.2)]])]
    rows, error = decompose(processes, [0, 0], ['a', 'b'], SETTINGS)
    assert all(r['advantage'] == 0 for r in rows)
    assert error == 0
    rows, error = decompose([process([[('duplicate', -.2)]]), process([[('duplicate', -.2)]])],
                            [1, 1], ['a', 'a'], SETTINGS)
    assert all(r['return_std'] == 0 and r['advantage'] == 0 for r in rows)
    assert error == 0


def test_difference_keeps_critical_fields_but_honors_normalization():
    d = argument_differences({'reservation_id': 'B', 'n': '1.0', 'flights': [{'id': 2}], 'empty': None},
                             {'reservation_id': 'A', 'n': 1, 'flights': [{'id': 1}], 'payment_id': 'P'})
    assert d == {'changed': ['flights', 'reservation_id'], 'missing': ['payment_id'], 'extra': []}


def test_task_centering_detects_between_task_confounding():
    rows = [{'task_id': t, 'outcome': y, 'feature': x}
            for t, x, ys in [('a', 0, [0, 0, 1]), ('b', 1, [0, 1, 1])] for y in ys]
    result = association(rows, 'feature')
    assert result['rho'] > 0
    assert result['within_task_centered_rho'] is None


def test_tool_types_come_from_official_decorators(tmp_path):
    p = tmp_path / 'tools.py'
    p.write_text('@is_tool(ToolType.READ)\ndef query(): pass\n\ndef helper(): pass\n')
    assert official_tool_types(p) == {'query': 'READ'}


def test_execution_comparison_distinguishes_ignored_fields_from_wrong_targets():
    gold = {'name': 'update_reservation_flights', 'arguments': {
        'reservation_id': 'A', 'cabin': 'economy', 'payment_id': 'P',
        'flights': [{'flight_number': 'F1', 'date': '2026-09-18'}]}}
    actual = {**gold, 'error': False, 'arguments': {**gold['arguments'],
              'flights': [{'flight_number': 'F1', 'date': '2026-09-18', 'price': 999}]}}
    result = execution_comparison(actual, [gold], [{**gold, 'error': True}, actual])
    assert result['execution_equivalent_gold_indices'] == [0]
    assert result['prior_equivalent_success_call_indices'] == [1]
    wrong = {**actual, 'arguments': {**actual['arguments'], 'reservation_id': 'B'}}
    assert execution_comparison(wrong, [gold], [actual])['execution_equivalent_gold_indices'] == []


def test_cli_refuses_selection_manifest(tmp_path):
    import hashlib
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text(json.dumps({'task_id': 'a', 'split': 'selection'}))
    report = tmp_path / 'report.json'
    report.write_text(json.dumps({'train_manifest': {'path': str(manifest),
                                                   'sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}}))
    with pytest.raises(ValueError, match='training-only'):
        main(['--irc-report', str(report), '--output-dir', str(tmp_path / 'output')])
    assert not (tmp_path / 'output').exists()


def test_duplicate_read_after_write_is_not_assumed_unchanged():
    def event(name, tier, obs):
        return {'name': name, 'arguments': {'id': 'A'}, 'error': False,
                'reward_type': tier, 'observation': obs}
    query = event('query', 'read_only', '{"status":"active"}')
    p = {'turn_records': [
        {'turn_index': 0, 'tool_calls': [query]},
        {'turn_index': 1, 'tool_calls': [event('cancel', 'gold_exact', 'ok')]},
        {'turn_index': 2, 'tool_calls': [event('query', 'duplicate', '{"status":"cancelled"}')]},
        {'turn_index': 3, 'tool_calls': [event('query', 'duplicate', '{ "status" : "cancelled" }')]},
    ]}
    history = duplicate_history(p, {'query': 'READ', 'cancel': 'WRITE'})
    assert history[2][0]['response_changed'] is True
    assert history[2][0]['intervening_successful_writes'] == 1
    assert history[3][0]['response_changed'] is False
    assert history[3][0]['intervening_successful_writes'] == 0

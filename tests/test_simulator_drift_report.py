from tau3_grpo.analysis.simulator_drift_report import normalize_review, recover_packet


def test_format_recovery_keeps_values_and_rejects_overlapping_objects():
    raw = {'raw_response': {'content': '{"coverage":"complete","findings":[]},"uncertainties":[],"summary":"x"}'}}
    packet, changes = recover_packet(raw)
    assert packet == {'coverage': 'complete', 'findings': [], 'uncertainties': [], 'summary': 'x'}
    assert changes
    raw['raw_response']['content'] = '{"coverage":"complete","findings":[]},"findings":[1]}'
    assert recover_packet(raw)[0] is None


def test_reference_expansion_does_not_repair_future_causality_or_wrong_role():
    payload = {'scenario_source_lines': [{'id': 's000'}],
        'contract_index': {'clauses': [{'id': 'C1', 'source_refs': ['s000']}]},
        'events': [{'event_id': 'm000', 'role': 'user', 'content': 'cancel'},
                   {'event_id': 'm001', 'role': 'assistant', 'content': 'done'}]}
    f = dict(kind='new_goal', support='direct', severity='major', event_id='m000',
             user_quote='cancel', scenario_refs=['C1'], prior_event_refs=['m001'],
             consequence='unknown', outcome_refs=[])
    raw = {'coverage': 'complete', 'findings': [f], 'uncertainties': []}
    result, changes, rejected = normalize_review(raw, payload)
    assert changes and rejected and result['coverage'] == 'incomplete'
    assert not result['findings']
    assert raw['findings'][0]['scenario_refs'] == ['C1']


def test_missing_metadata_does_not_become_a_complete_negative():
    payload = {'scenario_source_lines': [], 'contract_index': {'clauses': []}, 'events': []}
    result, _, _ = normalize_review({'coverage': 'complete', 'findings': []}, payload)
    assert result['coverage'] == 'incomplete'


def test_skeptical_review_cannot_omit_candidates_or_invent_refs():
    import pytest
    from tau3_grpo.analysis.simulator_drift_followup import validate_verification
    payload = {'scenario_source_lines': [{'id': 's000'}], 'events': [{'event_id': 'm001'}]}
    packet = {'decisions': [{'finding_index': 0, 'verdict': 'ambiguous', 'reason': 'Insufficient evidence',
                             'scenario_refs': ['s000'], 'event_refs': ['m001']}]}
    validate_verification(packet, [{}], payload)
    with pytest.raises(ValueError, match='Missing verification'):
        validate_verification(packet, [{}, {}], payload)
    packet['decisions'][0]['event_refs'] = ['m999']
    with pytest.raises(ValueError, match='event refs'):
        validate_verification(packet, [{}], payload)

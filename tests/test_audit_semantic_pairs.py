import asyncio
from copy import deepcopy
import json

import httpx
import pytest

from tau3_grpo.analysis.audit_semantic_pairs import (
    make_request, validate, combine, clique_verified, load_inputs, prefilter, run, evidence_payload,
)
from tau3_grpo.utils.hashing import sha256_file


def case(cid='a', text='Cancel ABC123.'):
    return {'id': cid, 'task_id': 't', 'db_hash': 'd', 'policy_hash': 'p',
            'comparison_group': 'g', 'messages': [{'role': 'user', 'content': text}]}


def response():
    return {'verdict': 'merge', 'shared_state': ['Cancel ABC123'], 'differences': [],
            'evidence': ['A:m0:s0', 'B:m0:s0']}


def inputs(tmp_path):
    cases = tmp_path/'cases.json'
    cases.write_text(json.dumps({'cases': [case(),case('b')]}))
    pairs = tmp_path/'pairs.json'
    pairs.write_text(json.dumps({'cases_sha256':sha256_file(cases),'pairs':[{'id':'p','a':'a','b':'b'}]}))
    config = {'provider':'openai_compatible','model':'fixture','cohort_mode':'evaluation_trials',
              'concurrency':2,'max_calls':2}
    return config, cases, pairs


def test_request_strips_labels_outcomes_and_hidden_fields():
    a=case(); a.update(expected='merge',reward=1,hidden_goal='SECRET')
    a['messages'][0].update(reward=1,hidden_goal='SECRET')
    request=make_request(a,case('b'))
    assert 'SECRET' not in json.dumps(request)
    assert 'reward' not in json.dumps(request['visible_messages'])
    assert set(request['visible_messages'])=={'A','B'}


@pytest.mark.parametrize('mutation', ['extra','verdict','empty','wrong_quote','negative_index','bool_index','wrong_side','one_side','merge_difference','separate_empty'])
def test_invalid_evidence_and_results_rejected(mutation):
    value=response()
    if mutation=='extra':value['extra']=1
    elif mutation=='verdict':value['verdict']='merge-ish'
    elif mutation=='empty':value['evidence']=[]
    elif mutation=='wrong_quote':value['evidence'][0]='A:m0:s999'
    elif mutation=='negative_index':value['evidence'][0]='A:m-1:s0'
    elif mutation=='bool_index':value['evidence'][0]=True
    elif mutation=='wrong_side':value['evidence'][0]='C:m0:s0'
    elif mutation=='one_side':value['evidence']=value['evidence'][:1]
    elif mutation=='merge_difference':value['differences']=['different payment']
    elif mutation=='separate_empty':value['verdict']='separate'
    with pytest.raises(ValueError):validate(value,make_request(case(),case('b'))['visible_messages'])


def test_abstention_no_evidence_and_conservative_symmetry():
    validate({'verdict':'abstain','shared_state':[],'differences':['unclear'],'evidence':[]}, {})
    assert combine({'verdict':'merge'}, {'verdict':'separate'})=='abstain'
    assert combine({'verdict':'merge'}, {'verdict':'merge','error':'bad quote'})=='abstain'
    assert combine({'verdict':'merge'}, {'verdict':'merge'})=='merge'


def test_evidence_ids_preserve_every_character_and_resolve_exactly():
    text='I’m sure.\n\n'+('x'*1500)+'\r\nFinal line.'
    histories=make_request(case(text=text),case('b'))['visible_messages']
    payload,catalogue=evidence_payload(histories)
    segments=payload['histories']['A'][0]['content_segments']
    assert ''.join(s['text'] for s in segments)==text
    for s in segments:
        e=catalogue[s['ref']]
        assert text[e['start']:e['end']]==e['quote']==s['text']
    value=response();value['evidence']=['A:m0:s0','A:m0:s0']
    with pytest.raises(ValueError):validate(value,histories)


def test_grouping_requires_all_edges_not_transitive_closure():
    relations={frozenset(('a','b')):'merge',frozenset(('b','c')):'merge'}
    assert not clique_verified(['a','b','c'],relations)
    relations[frozenset(('a','c'))]='separate'
    assert not clique_verified(['a','b','c'],relations)
    relations[frozenset(('a','c'))]='merge'
    assert clique_verified(['a','b','c'],relations)


@pytest.mark.parametrize('field',['task_id','comparison_group','policy_hash','db_hash'])
def test_cross_scope_is_never_model_merge(field):
    a,b=case(),case('b'); b[field]='different'
    assert prefilter(a,b)=='scope_difference:'+field
    b.pop(field)
    with pytest.raises(ValueError):prefilter(a,b)


def test_preflight_rejects_labels_and_budget_before_api(tmp_path):
    config,cases,pairs=inputs(tmp_path)
    with pytest.raises(ValueError,match='over_budget'):load_inputs({**config,'max_calls':1},cases,pairs)
    data=json.loads(pairs.read_text());data['pairs'][0]['expected']='merge';pairs.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='labels'):load_inputs(config,cases,pairs)


def test_remote_mock_transport_raw_capture_and_two_directions(monkeypatch,tmp_path):
    config,cases,pairs=inputs(tmp_path)
    monkeypatch.setenv('TAU3_ENV_FILE',str(tmp_path/'missing'))
    monkeypatch.setenv('TAU3_SEMANTIC_API_KEY','test')
    monkeypatch.setenv('TAU3_SEMANTIC_BASE_URL','https://example.com/v1')
    calls=[]
    def handler(req):
        body=json.loads(req.content);payload=json.loads(body['messages'][1]['content'])
        assert set(payload)=={'schema','histories'}
        assert 'expected' not in json.dumps(payload)
        calls.append(payload)
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(response())}}]})
    out=tmp_path/'out'
    summary=asyncio.run(run(config,cases,pairs,out,transport=httpx.MockTransport(handler)))
    assert summary['requests']==len(calls)==2 and summary['verdicts']['merge']==1
    assert len((out/'raw_responses.jsonl').read_text().splitlines())==2
    with pytest.raises(FileExistsError):asyncio.run(run(config,cases,pairs,out,transport=httpx.MockTransport(handler)))
    assert len(calls)==2

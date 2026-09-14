"""Replay safety checks using the real official Airline environment and tools."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from tau3_grpo.analysis.replay_decisions import normalized_content, replay_messages, run
from tau3_grpo.envs.adapter import build_environment, load_default_flight_db
from tau3_grpo.envs.tau2_bridge import message_models
from tau3_grpo.envs.registry import SessionEntry
from tau3_grpo.launch import prepare
from tau3_grpo.paths import CODE_ROOT


@pytest.fixture(scope='module')
def database():return load_default_flight_db()


def env(database):return build_environment(database)


def record(environment,calls):
    return [environment.get_response(message_models()['ToolCall'].model_validate(c)).model_dump(mode='json') for c in calls]


def history(database):
    rid=next(rid for rid,r in database.reservations.items() if r.status!='cancelled')
    calls=[{'id':'cancel','name':'cancel_reservation','arguments':{'reservation_id':rid},'requestor':'assistant'},
           {'id':'read','name':'get_reservation_details','arguments':{'reservation_id':rid},'requestor':'assistant'}]
    return [{'role':'user','content':'Please cancel the reservation.'},
            {'role':'assistant','content':None,'tool_calls':calls},
            *record(env(database),calls),{'role':'assistant','content':'It is cancelled.'}]


def test_real_write_then_read_executes_entire_batch_in_order(database):
    original=env(database).get_db_hash();h=history(database)
    r=replay_messages(env(database),h,task_id='t')
    assert r['status']=='verified' and len(r['events'])==2
    assert [e['tool_name'] for e in r['events']]==['cancel_reservation','get_reservation_details']
    assert r['snapshots'][0]['db_hash']==original
    assert r['snapshots'][1]['db_hash']!=original
    assert r['events'][0]['db_after']==r['events'][1]['db_before']==r['snapshots'][1]['db_hash']
    assert env(database).get_db_hash()==original  # Both runs had isolated DBs.


def test_mismatch_stops_later_state_certification(database):
    h=history(database);h[2]['content']='{"fabricated":true}'
    r=replay_messages(env(database),h,task_id='t')
    assert r['failure']['kind']=='tool_result_mismatch'
    assert len(r['snapshots'])==1 and len(r['events'])==0
    assert r['snapshots'][0]['verified_tool_responses']==0


@pytest.mark.parametrize('change,kind', [('swap','unexpected_tool_response_order'),('drop','missing_tool_response'),('duplicate','duplicate_or_missing_call_id')])
def test_incomplete_and_ambiguous_batches_fail_closed(database,change,kind):
    h=history(database)
    if change=='swap':h[2],h[3]=h[3],h[2]
    elif change=='drop':h=h[:2]
    else:h[1]['tool_calls'][1]['id']='cancel'
    r=replay_messages(env(database),h,task_id='t')
    assert r['failure']['kind']==kind
    assert len(r['snapshots'])==1


def test_current_action_and_future_text_do_not_enter_pre_action_anchor(database):
    a=history(database);b=deepcopy(a)
    b[1]['content']='A completely different future explanation.'
    b[1]['tool_calls'][0]['arguments']['reservation_id']='does-not-exist'
    b[-1]['content']='A future outcome cannot leak into the first snapshot.'
    ra=replay_messages(env(database),a,task_id='t');rb=replay_messages(env(database),b,task_id='t')
    assert ra['snapshots'][0]==rb['snapshots'][0]
    assert rb['failure'] is not None


def test_matching_error_response_replays_but_error_flag_mismatch_rejected(database):
    calls=[{'id':'e','name':'get_reservation_details','arguments':{'reservation_id':'not-a-real-reservation'},'requestor':'assistant'}]
    h=[{'role':'assistant','content':None,'tool_calls':calls},*record(env(database),calls),{'role':'assistant','content':'Need another ID.'}]
    assert h[1]['error'] is True
    assert replay_messages(env(database),h,task_id='t')['status']=='verified'
    h[1]['error']=False
    assert replay_messages(env(database),h,task_id='t')['failure']['kind']=='tool_result_mismatch'


def test_policy_mismatch_prevents_even_initial_state_certification(database):
    r=replay_messages(env(database),history(database),task_id='t',expected_policy='other policy')
    assert r['status']=='unverified' and not r['snapshots']


def test_comparison_never_erases_array_order_bool_or_numeric_representation():
    assert normalized_content('{"a":1,"b":2}','json')==normalized_content('{ "b":2, "a":1 }','json')
    for a,b in [('[1,2]','[2,1]'),('true','1'),('25','25.0')]:
        assert normalized_content(a,'json')!=normalized_content(b,'json')
    assert normalized_content(' {"a":1}','exact_text')!=normalized_content('{"a":1}','exact_text')


def test_disabled_replay_does_not_read_inputs_or_create_output(tmp_path):
    target=tmp_path/'absent'
    assert run({'enabled':False},['missing-source'],target)['enabled'] is False
    assert not target.exists()


@pytest.mark.parametrize('version',['v1','v2','v3','v4'])
def test_switch_reaches_shell_ray_and_metadata_without_starting_run(tmp_path,version):
    profile=CODE_ROOT/'configs/train/rl/qwen35_4b_full_a800_c50_matched6h_e2_20260912.yaml' if version=='v1' else CODE_ROOT/f'configs/train/rl/qwen35_4b_full_a800_anchor_{version}_20260914.yaml'
    inherited={**os.environ,'TAU3_ROOT':str(tmp_path),'TAU3_RUN_ROOT':str(tmp_path/'runs'),
               'TAU3_ENV_FILE':str(tmp_path/'absent.env'),'TAU3_DRY_RUN':'1',
               'PATH':str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']}
    inherited.pop('TAU3_GRPO_ANCHOR_VERSION',None)
    command,values,_=prepare('rl',profile,'e2',42,[],inherited)
    result=subprocess.run(command,env=values,cwd=CODE_ROOT,text=True,capture_output=True,check=True,timeout=20)
    rendered=' '.join(shlex.split(result.stdout.splitlines()[-1]))
    assert f'anchor_version:{version}' in rendered
    assert f'runtime_env.env_vars.TAU3_GRPO_ANCHOR_VERSION={version}' in rendered
    assert 'episode_normalization' not in rendered  # This change isolates anchors.
    assert not (tmp_path/'runs').exists()
    if version=='v1':assert SessionEntry(session=None).anchor_version=='v1'

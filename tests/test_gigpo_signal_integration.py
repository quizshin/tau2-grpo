"""CPU checks at the real veRL compute_advantage/filter boundary."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage, tau3_dynamic_filter
from tau3_grpo.algorithms.verl_estimator import compute_tau_gigpo_verl
from tau3_grpo.tracking.signal_audit import analyze, audit_update, capture_mask, restore_mask


def object_rows(values):
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


def batch_fixture(dtype=torch.float64, all_success=False):
    lengths = list(range(2, 10))
    rewards = torch.zeros(8, 10, dtype=dtype)
    rewards[:, -1] = torch.tensor([1] * 8 if all_success else [1,0,0,0,1,0,0,0], dtype=dtype)
    mask = torch.tensor([[int(j < length and j != 1) for j in range(10)] for length in lengths])
    return DataProto.from_dict(tensors={'token_level_rewards': rewards, 'response_mask': mask}, non_tensors={
        'uid': np.array(['task']*8), 'task_id': np.array(['task']*8),
        'anchor_ids': object_rows([['initial'] + [f'{i}-{j}' for j in range(1,n)] for i,n in enumerate(lengths)]),
        'anchor_spans': object_rows([[[j,j+1] for j in range(n)] for n in lengths]),
    })


def cfg(omega=1., norm='grpo', df=False):
    return OmegaConf.create({'adv_estimator': 'tau_gigpo', 'norm_adv_by_std_in_grpo': True,
                            'gigpo': {'episode_normalization': norm, 'omega': omega},
                            'dynamic_filter': {'enable': df, 'group_size': 8}})


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('normalize', [True, False])
@pytest.mark.parametrize('success', [True, False])
def test_omega_zero_matches_real_trainer_both_outputs(dtype, normalize, success):
    data = batch_fixture(dtype, success)
    expected = compute_advantage(deepcopy(data), 'grpo', norm_adv_by_std_in_grpo=normalize)
    config = cfg(0)
    # Explicit caller argument must win over config (same contract as GRPO).
    actual = compute_advantage(data, 'tau_gigpo', norm_adv_by_std_in_grpo=normalize, config=config)
    assert torch.equal(actual.batch['advantages'], expected.batch['advantages'])
    assert torch.equal(actual.batch['returns'], expected.batch['returns'])


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_omega_zero_empty_mask_and_singleton(dtype):
    data = batch_fixture(dtype)
    data.non_tensor_batch['uid'] = np.array([str(i) for i in range(8)])
    data.batch['response_mask'][0] = 0
    expected = compute_advantage(deepcopy(data), 'grpo')
    actual = compute_advantage(data, 'tau_gigpo', config=cfg(0))
    assert torch.equal(actual.batch['advantages'], expected.batch['advantages'])
    assert torch.equal(actual.batch['returns'], expected.batch['returns'])


def test_legacy_default_unchanged_and_invalid_mode_fails():
    data = batch_fixture()
    kwargs = dict(token_level_rewards=data.batch['token_level_rewards'], response_mask=data.batch['response_mask'],
                  index=data.non_tensor_batch['uid'], non_tensor_batch=data.non_tensor_batch)
    a,_ = compute_tau_gigpo_verl(**kwargs)
    b,_ = compute_tau_gigpo_verl(**kwargs, config=cfg(norm='legacy_mean'))
    assert torch.equal(a,b)
    with pytest.raises(ValueError, match='episode_normalization'):
        compute_tau_gigpo_verl(**kwargs, config=cfg(norm='typo'))


def test_real_filter_audit_payload_replays_without_changing_batch(tmp_path, monkeypatch):
    monkeypatch.setenv('TAU3_GRPO_SIGNAL_AUDIT_DIR', str(tmp_path))
    data = batch_fixture(all_success=True); config=cfg(df=True)
    before = capture_mask(data, config)
    tau3_dynamic_filter(data, config)
    compute_advantage(data, 'tau_gigpo', config=config)
    frozen = deepcopy(data)
    metrics = audit_update(data, before, config, 1)
    assert metrics['gigpo_signal/before_nonzero_steps'] == 8
    assert metrics['gigpo_signal/after_nonzero_steps'] == 0
    assert metrics['gigpo_signal/masked_nonzero_steps'] == 8
    for key in frozen.batch.keys(): assert torch.equal(frozen.batch[key],data.batch[key])
    payload=json.loads(next(tmp_path.glob('*.json')).read_text())
    length=payload['response_length']; rows=payload['rows']
    assert np.array_equal(restore_mask([r['mask_before'] for r in rows],length),before)
    rewards=torch.zeros(8,length,dtype=torch.float64)
    for i,row in enumerate(rows):
        for j,value in row['reward_entries']: rewards[i,j]=value
    replay,_=compute_tau_gigpo_verl(rewards,torch.as_tensor(restore_mask([r['mask_after'] for r in rows],length)),
        index=[r['metadata']['uid'] for r in rows], config=OmegaConf.create(payload['algorithm']),
        non_tensor_batch={'anchor_ids':[r['anchor_ids'] for r in rows], 'anchor_spans':[r['anchor_spans'] for r in rows]})
    assert hashlib.sha256(replay.numpy().tobytes()).hexdigest()==payload['actual_advantage_sha256']
    audit_update(data,before,config,1)
    assert len(list(tmp_path.glob('*.json')))==2  # retry evidence retained


def test_audit_disabled_and_grpo_do_not_capture(monkeypatch):
    data=batch_fixture()
    monkeypatch.delenv('TAU3_GRPO_SIGNAL_AUDIT_DIR',raising=False)
    assert capture_mask(data,cfg()) is None
    assert audit_update(data,None,cfg(),1)=={}
    monkeypatch.setenv('TAU3_GRPO_SIGNAL_AUDIT_DIR','/unused')
    assert capture_mask(data,{'adv_estimator':'grpo'}) is None


def test_mixed_cross_noninitial_signal_and_omega_zero():
    data=batch_fixture(); config=cfg()
    data.non_tensor_batch['anchor_ids']=object_rows([['initial','later']]*8)
    data.non_tensor_batch['anchor_spans']=object_rows([[[0,1],[2,3]]]*8)
    before=data.batch['response_mask'].numpy().copy()
    compute_advantage(data,'tau_gigpo',config=config)
    metrics=analyze(data,before,config)['metrics']
    assert metrics['after_cross_noninitial_nonzero_steps']==7  # shortest row has no token 2
    assert metrics['after_weighted_abs_step_advantage']>0
    config=cfg(omega=0);compute_advantage(data,'tau_gigpo',config=config)
    metrics=analyze(data,before,config)['metrics']
    assert metrics['after_weighted_abs_step_advantage']==0
    assert metrics['after_nonzero_steps']>0  # raw term, documented explicitly


def test_candidate_profile_dry_run(tmp_path):
    from tau3_grpo.launch import prepare
    from tau3_grpo.paths import CODE_ROOT
    command,env,_=prepare('rl',CODE_ROOT/'configs/train/rl/qwen35_4b_full_a800_gigpo_audit_20260914.yaml',
                          'e2',42,[],{'TAU3_ROOT':str(tmp_path),'TAU3_RUN_ROOT':str(tmp_path/'runs')})
    assert '++algorithm.gigpo.episode_normalization=grpo' in command
    assert 'trainer.total_training_steps=2' in command
    assert env['TAU3_GRPO_SIGNAL_AUDIT_DIR'].endswith('/e2_seed42/signal-audits')
    assert not (tmp_path/'runs').exists()


def test_nonzero_payload_replay_and_manual_step_term():
    data=batch_fixture();config=cfg(omega=0.7)
    before=data.batch['response_mask'].numpy().copy()
    baseline=compute_advantage(deepcopy(data),'grpo').batch['advantages'].clone()
    compute_advantage(data,'tau_gigpo',config=config)
    # Only the shared initial anchor has a nonzero step term in this fixture.
    returns=np.array([1,0,0,0,1,0,0,0],dtype=float)
    raw=returns * (0.95 ** np.arange(1,9))
    expected=baseline.clone();expected[:,0]+=torch.tensor(0.7*(raw-raw.mean()))
    assert torch.allclose(data.batch['advantages'],expected,atol=1e-12,rtol=0)
    payload=analyze(data,before,config)
    replay_rewards=torch.zeros_like(data.batch['token_level_rewards'])
    for i,row in enumerate(payload['rows']):
        for j,v in row['reward_entries']:replay_rewards[i,j]=v
    replay,_=compute_tau_gigpo_verl(replay_rewards,
        torch.as_tensor(restore_mask([r['mask_after'] for r in payload['rows']],payload['response_length'])),
        index=[r['metadata']['uid'] for r in payload['rows']],config=OmegaConf.create(payload['algorithm']),
        non_tensor_batch={key:[r[key] for r in payload['rows']] for key in ('anchor_ids','anchor_spans')})
    assert torch.count_nonzero(replay)>0
    assert hashlib.sha256(replay.numpy().tobytes()).hexdigest()==payload['actual_advantage_sha256']


def test_padding_is_excluded_and_corrupt_padding_fails():
    data=batch_fixture();config=cfg()
    data.non_tensor_batch['tau3_is_padding']=np.array([False]*7+[True])
    data.non_tensor_batch['anchor_ids'][-1]=[];data.non_tensor_batch['anchor_spans'][-1]=[]
    data.non_tensor_batch['uid'][-1]='pad'
    data.batch['response_mask'][-1]=0
    before=data.batch['response_mask'].numpy().copy()
    compute_advantage(data,'tau_gigpo',config=config)
    payload=analyze(data,before,config)
    assert payload['metrics']['real_trajectories']==7
    assert all(s['trajectory']!=7 for s in payload['steps'])
    before[-1,0]=1
    with pytest.raises(ValueError,match='padding'):
        analyze(data,before,config)


def test_dialogue_screening_excludes_stop_control_messages(tmp_path):
    from env_info.a800_20260912.dialogue_anchor_audit import audit
    p=tmp_path/'trajectories.jsonl'
    p.write_text(json.dumps({'task_id':'t','trial':0,'simulation':{'messages':[
        {'role':'assistant','content':'Search first?'},
        {'role':'user','content':'Yes, search first.'},
        {'role':'assistant','content':'Done'},
        {'role':'user','content':'###STOP###'},
        {'role':'user','content':'Thanks, goodbye! ###STOP###'},
    ]}})+'\n')
    result=audit([p]);assert result['counts']['terminal_control_messages']==2
    assert result['counts']['user_turns']==2
    assert result['counts'].get('latched_after_pause_cue',0)==0


def test_batch_diagnostics_survive_later_estimator_calls_and_reset_on_switch():
    from tau3_grpo.integrations.verl.gigpo import last_stats
    from tau3_grpo.tracking.trainer_telemetry import collect_rollout_metrics

    first = compute_advantage(batch_fixture(), 'tau_gigpo', config=cfg())
    saved = deepcopy(first.meta_info['tau3_estimator_diagnostics'])
    assert saved['stats'] == last_stats()
    second = batch_fixture()
    second.batch['response_mask'].zero_()
    compute_advantage(second, 'tau_gigpo', config=cfg())
    assert first.meta_info['tau3_estimator_diagnostics'] == saved
    metrics = collect_rollout_metrics(first.non_tensor_batch, adv_estimator='tau_gigpo',
                                      estimator_diagnostics=saved)
    assert all(metrics[f'gigpo/{key}'] == value for key, value in saved['stats'].items())
    compute_advantage(first, 'grpo')
    assert first.meta_info['tau3_estimator_diagnostics'] == {'estimator': 'grpo', 'stats': {}}

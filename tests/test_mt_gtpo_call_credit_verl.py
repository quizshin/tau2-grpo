import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_mt_gtpo_call_credit import identity_batch

from tau3_grpo.algorithms.mt_gtpo import compute_mt_gtpo
from tau3_grpo.integrations.verl.mt_gtpo import compute_mt_gtpo_verl, settings_from_config


@pytest.mark.parametrize('filtering', [False, True])
@pytest.mark.parametrize('mode', ['call_local_v1', 'call_residual_v1'])
def test_native_trainer_adapter_records_actual_call_advantages(filtering, mode):
    from tensordict import TensorDict
    from verl.trainer.ppo.ray_trainer import compute_advantage

    from verl import DataProto

    value, ids, processes, facts = identity_batch()
    metadata = dict(uid=np.asarray(value['uids']),
                    process_reward_json=np.asarray(list(map(json.dumps, processes)), dtype=object),
                    trajectory_facts_json=np.asarray(list(map(json.dumps, facts)), dtype=object))
    data = DataProto(batch=TensorDict(dict(responses=torch.tensor(ids),
        response_mask=torch.tensor(value['response_mask']), token_level_rewards=torch.zeros((2, 11))),
        batch_size=[2]), non_tensor_batch=metadata)
    config = OmegaConf.create(dict(adv_estimator='mt_gtpo', mt_gtpo=dict(credit_mode=mode),
        dynamic_filter=dict(enable=filtering, group_size=2, metric='advantage')))
    compute_advantage(data, 'mt_gtpo', config=config)
    adv = data.batch['advantages']
    assert adv[0, 1] < 0 < adv[0, 3]
    assert torch.equal(data.batch['response_mask'], torch.tensor(value['response_mask']))
    for i, raw in enumerate(data.non_tensor_batch['mt_gtpo_replay_json']):
        row = json.loads(raw)
        assert row['schema'] == 'mt_gtpo_call_replay_v1'
        assert row['credit_mode'] == mode
        assert 'turn_advantages' not in row
        np.testing.assert_allclose(row['token_advantages'], adv[i].numpy(), rtol=1e-6)
        assert row['call_receipts'][0]['eligible']
    assert data.meta_info['tau3_estimator_diagnostics']['stats']['call_credit/applied_calls'] == (4 if mode == 'call_local_v1' else 3)
    stats = data.meta_info['tau3_estimator_diagnostics']['stats']
    assert stats['call_credit/error_calls'] == 2
    assert stats['call_credit/error_positive_before'] == 1
    assert stats['call_credit/error_positive_after'] == 0
    assert stats['call_credit/positive_reward_signal_lost'] == 0


@pytest.mark.parametrize('mode', [None, 'turn_v1'])
def test_old_mode_exactly_preserves_output_and_replay_schema(mode):
    value, _, processes, _ = identity_batch()
    config = {} if mode is None else {'mt_gtpo': {'credit_mode': mode}}
    metadata = {'process_reward_json': np.asarray(list(map(json.dumps, processes)), dtype=object)}
    adv, returns = compute_mt_gtpo_verl(torch.zeros((2, 11)), torch.tensor(value['response_mask']),
                                      value['uids'], config, metadata)
    old_adv, old_returns, _ = compute_mt_gtpo(value['outcomes'], value['uids'],
        value['turn_rewards'], value['turn_spans'], value['response_mask'])
    np.testing.assert_array_equal(adv.numpy(), old_adv.astype(np.float32))
    np.testing.assert_array_equal(returns.numpy(), old_returns.astype(np.float32))
    assert all(json.loads(raw)['schema'] == 'mt_gtpo_replay_v1' for raw in metadata['mt_gtpo_replay_json'])


@pytest.mark.parametrize('mode', ['call_local_v1', 'call_residual_v1'])
def test_candidate_requires_runtime_facts_and_actual_tokens(mode):
    value, _, processes, _ = identity_batch()
    with pytest.raises(ValueError, match='actual batch responses'):
        compute_mt_gtpo_verl(torch.zeros((2, 11)), torch.tensor(value['response_mask']), value['uids'],
            {'mt_gtpo': {'credit_mode': mode}},
            {'process_reward_json': np.asarray(list(map(json.dumps, processes)), dtype=object)})
    with pytest.raises(ValueError, match='credit mode'):
        settings_from_config({'mt_gtpo': {'credit_mode': 'typo'}})

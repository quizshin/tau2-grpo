"""CPU behavioral gates for isolated candidates, not production replacements."""
import numpy as np
import pytest
import torch

from env_info.a800_20260912.cpu_signal_validation import (
    ScopedConfirmation, StepRecord, aligned_candidate, all_success_fixture,
    apply_dynamic_filter, compute_tau_gigpo_advantage, confirmation_flags,
    cross_trajectory_steps, current_anchor, message, observation_fingerprint,
    read_messages, signal_stats, step_advantages, stock_grpo,
)


@pytest.mark.parametrize('rewards', [[1.]*4+[0.]*4, [1.]+[0.]*7, [1.]*8, [0.]*8])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_aligned_omega0_exactly_matches_real_grpo(rewards, dtype):
    mask=np.array([[1,1,0,1,0],[0,1,1,0,0]]*4,dtype=float)
    steps=[StepRecord(i,0,'state',(0,5)) for i in range(8)]
    baseline=stock_grpo(rewards,['task']*8,mask,dtype=dtype)
    candidate=aligned_candidate(rewards,['task']*8,mask,steps,omega=0,dtype=dtype)
    assert torch.equal(candidate,baseline)
    assert torch.isfinite(candidate).all() and not candidate.numpy()[mask==0].any()


def test_current_omega0_differs_from_e0_but_matches_mean_baseline():
    rewards=[1.]*4+[0.]*4;mask=np.ones((8,4))
    current,_=compute_tau_gigpo_advantage(rewards,['task']*8,[],response_length=4,
                                         response_mask=mask,omega=0)
    assert current[0,0]==.5
    assert stock_grpo(rewards,['task']*8,mask)[0,0].item()==pytest.approx(.9354126,abs=1e-6)
    np.testing.assert_array_equal(current,stock_grpo(rewards,['task']*8,mask,normalize=False).numpy())


def test_empty_mask_and_singleton_protocol_are_explicit():
    mask=np.zeros((8,3));steps=[StepRecord(i,0,'state',(0,3)) for i in range(8)]
    assert not aligned_candidate([1.]*4+[0.]*4,['task']*8,mask,steps,omega=1).any()
    # Stock veRL gives singleton reward a zero baseline; the current custom
    # estimator mean-centres a singleton to zero. Singleton real groups are not
    # used in the formal 8-way schedule, but the candidate must preserve stock behavior.
    singleton=aligned_candidate([1.],['single'],np.ones((1,2)),[],omega=0)
    assert torch.equal(singleton,stock_grpo([1.],['single'],np.ones((1,2))))


def test_step_only_changes_its_policy_span():
    mask=np.array([[1,0,1,1],[1,0,1,1]],dtype=float)
    steps=[StepRecord(i,0,'shared',(2,4)) for i in range(2)]
    base=stock_grpo([1.,0.],['task']*2,mask)
    result=aligned_candidate([1.,0.],['task']*2,mask,steps,omega=.2)
    expected=np.array([[0,0,.1,.1],[0,0,-.1,-.1]])
    np.testing.assert_allclose((result-base).numpy(),expected,atol=1e-12)


@pytest.mark.parametrize('text', ['I do not confirm this cancellation.', 'Yes, that is my name.',
                                  'What happens if I confirm?', 'Please wait, do not proceed.'])
def test_current_feature_false_positives_are_reproduced(text):
    assert 'user_confirmed' in confirmation_flags([message('user',text)])


ACTION={'operation':'cancel_reservation','reservation_id':'A','amount':100,'currency':'USD'}


@pytest.mark.parametrize('text,expected', [('Yes.','confirmed'),('Yes, please go ahead.','confirmed'),
    ('I do not confirm this cancellation.','denied'),('Yes, that is my name.','unknown'),
    ('What happens if I confirm?','unknown'),('Please wait, do not proceed.','denied')])
def test_scoped_fixture_confirmation(text,expected):
    tracker=ScopedConfirmation();tracker.propose(ACTION)
    assert tracker.reply(text)==expected
    assert tracker.authorizes(ACTION)==(expected=='confirmed')


@pytest.mark.parametrize('change', [{'reservation_id':'B'},{'amount':200},{'currency':'EUR'},
                                   {'operation':'update_reservation_flights'}])
def test_confirmation_invalidated_when_scope_changes(change):
    tracker=ScopedConfirmation();tracker.propose(ACTION);tracker.reply('Yes.')
    assert tracker.authorizes(ACTION)
    new={**ACTION,**change};tracker.propose(new)
    assert not tracker.authorizes(new) and not tracker.authorizes(ACTION)


def test_withdrawal_and_session_isolation():
    one,two=ScopedConfirmation(),ScopedConfirmation()
    one.propose(ACTION);one.reply('Yes.')
    assert one.authorizes(ACTION) and not two.authorizes(ACTION)
    one.reply('Wait, I withdraw my consent.');assert not one.authorizes(ACTION)
    two.reply('Yes.');assert not two.authorizes(ACTION)


@pytest.mark.parametrize('other',[('B',100),('A',200)])
def test_read_observation_collisions_and_conservative_candidate(other):
    a,b=read_messages('A',100),read_messages(*other)
    assert current_anchor(a)==current_anchor(b)
    assert observation_fingerprint(a)!=observation_fingerprint(b)
    assert observation_fingerprint(a)==observation_fingerprint(read_messages('A',100))


def test_read_fingerprint_ignores_json_key_order_and_call_ids():
    a,b=read_messages(),read_messages()
    b[0].tool_calls[0].id='changed';b[1].id='changed'
    b[1].content='{"price":100,"reservation_id":"A"}'
    assert observation_fingerprint(a)==observation_fingerprint(b)


def test_all_success_filter_removes_nonzero_discount_signal():
    steps,mask=all_success_fixture();rewards=[1.]*8
    values,_=step_advantages(steps,rewards)
    after,_=apply_dynamic_filter(mask,rewards,['task']*8)
    stats=signal_stats(steps,values,mask,after)
    assert stats['nonzero_steps_before']==8 and stats['nonzero_steps_after']==0
    assert stats['abs_step_advantage_mass_before']>0 and stats['abs_step_advantage_mass_after']==0
    assert stats['cross_trajectory_noninitial_nonzero_steps']==0
    gamma1,_=step_advantages(steps,rewards,gamma=1.)
    assert not gamma1.any()


def test_zero_reward_cannot_create_step_signal():
    steps,mask=all_success_fixture();values,stats=step_advantages(steps,[0.]*8)
    assert stats.usable_step_coverage>0 and not values.any()


def test_self_repeated_state_is_not_cross_trajectory_evidence():
    steps=[StepRecord(0,j,'state',(j,j+1)) for j in range(2)]
    old,_=step_advantages(steps,[1.]);np.testing.assert_allclose(old,[-.025,.025])
    assert not cross_trajectory_steps(steps,[1.]).any()


def test_mixed_group_preserves_noninitial_cross_trajectory_signal():
    steps=[StepRecord(i,j,'initial' if j==0 else 'later',(j*2,j*2+2)) for i in range(8) for j in range(2)]
    rewards=[1.]*4+[0.]*4;mask=np.ones((8,4));values,_=step_advantages(steps,rewards)
    after,_=apply_dynamic_filter(mask,rewards,['task']*8);stats=signal_stats(steps,values,mask,after)
    assert stats['nonzero_steps_before']==stats['nonzero_steps_after']==16
    assert stats['cross_trajectory_noninitial_nonzero_steps']==8
    assert stats['nonzero_tokens_before']==stats['nonzero_tokens_after']==32
    assert stats['abs_step_advantage_mass_before']==stats['abs_step_advantage_mass_after']


def test_no_cuda_context_was_initialized():
    assert not torch.cuda.is_initialized()

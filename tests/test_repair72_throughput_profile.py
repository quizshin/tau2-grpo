"""Execution tuning must not silently reduce the formal learning/eval budget."""
from pathlib import Path

from tau3_grpo.configuration import load_config


def test_throughput_profile_preserves_experiment_and_only_changes_execution():
    directory = Path(__file__).resolve().parents[1] / 'configs/train/rl'
    old = load_config(directory / 'repair72_grpo_2xa800_passenger_v1.yaml')['launch']
    new = load_config(directory / 'repair72_grpo_2xa800_passenger_v1_throughput.yaml')['launch']
    changed = {k for k in new['environment'] if new['environment'][k] != old['environment'].get(k)}
    assert changed == {'MAX_NUM_SEQS', 'TAU3_USER_MAX_NUM_SEQS', 'TAU3_EXPERIMENT_LABEL'}
    assert new['overrides'][:len(old['overrides'])] == old['overrides']
    assert new['overrides'][len(old['overrides']):] == [
        'actor_rollout_ref.actor.fsdp_config.param_offload=false',
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload=false',
        'actor_rollout_ref.rollout.gpu_memory_utilization=0.50',
    ]
    assert new['terminal_reward_protocol'] == old['terminal_reward_protocol']
    for k in ('GROUP_SIZE', 'GROUPS_PER_UPDATE', 'TOTAL_UPDATES', 'MAX_RESPONSE_LENGTH',
              'MAX_MODEL_LENGTH', 'MAX_ASSISTANT_TURNS', 'MAX_USER_TURNS', 'MODEL_PATH',
              'TAU3_POLICY_CUDA_DEVICES', 'TAU3_USER_CUDA_DEVICES'):
        assert new['environment'][k] == old['environment'][k]

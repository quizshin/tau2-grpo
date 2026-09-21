import yaml

from tau3_grpo.envs.simulator_config import main


def test_external_qwen38_alias_can_disable_thinking_without_policy_detection(tmp_path):
    path = tmp_path / 'interaction.yaml'
    main(['--output', str(path), '--model', 'Qwen/Qwen3.8-27B-AWQ-INT4',
          '--thinking', 'off', '--base-url', 'http://localhost:8101/v1'])
    config = yaml.safe_load(path.read_text())['interaction'][0]['config']
    assert config['user_temperature'] == 0.7
    assert config['user_base_url'] == 'http://localhost:8101/v1'
    assert config['user_llm_args']['extra_body']['chat_template_kwargs'] == {'enable_thinking': False}


def test_legacy_simulator_auto_does_not_inject_thinking(tmp_path):
    path = tmp_path / 'interaction.yaml'
    main(['--output', str(path), '--model', 'Qwen/Qwen2.5-7B-Instruct'])
    config = yaml.safe_load(path.read_text())['interaction'][0]['config']
    assert not config.get('user_llm_args', {}).get('extra_body', {}).get('chat_template_kwargs')


def test_runtime_defaults_reach_actual_request_args():
    from tau3_grpo.envs.interaction import Tau3AirlineInteraction
    from tau3_grpo.envs.session import UserSimulatorConfig
    from tau3_grpo.evaluation.runtime import Endpoint

    assert Endpoint('policy', 'http://unused').llm_args()['temperature'] == 0.7
    assert UserSimulatorConfig('user').to_llm_args()['temperature'] == 0.7
    assert Tau3AirlineInteraction({})._user_config.to_llm_args()['temperature'] == 0.7

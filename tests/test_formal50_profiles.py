import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize('arm', ['e0', 'e1', 'e2', 'e3'])
def test_matched_profile_fixed_sampling_and_ray_budget(arm, tmp_path, monkeypatch):
    monkeypatch.setenv('TAU3_ROOT', str(tmp_path))
    monkeypatch.setenv('TAU3_RUN_ROOT', str(tmp_path / 'runs'))
    monkeypatch.setenv('TAU3_MODEL_ROOT', str(tmp_path / 'models'))
    monkeypatch.setenv('TAU3_ENV_FILE', str(tmp_path / 'absent.env'))
    monkeypatch.setenv('PATH', str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
    monkeypatch.setenv('GROUPS_PER_UPDATE', '16')
    monkeypatch.setenv('TOTAL_UPDATES', '40')
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'env_info/a800_20260912'))
    prepare = importlib.import_module('prepare_formal50')
    importlib.reload(prepare)
    run = importlib.import_module('run_matched50')
    importlib.reload(run)
    command, env, snapshot = run.stage_config(arm, 100 if arm == 'e0' else 30, 12345)
    output = subprocess.check_output(command, env=dict(env, TAU3_DRY_RUN='1'), text=True)
    args = shlex.split(output.splitlines()[-1])
    settings = dict(arg.lstrip('+').split('=', 1) for arg in args if '=' in arg)
    assert env['GROUPS_PER_UPDATE'] == env['GROUP_SIZE'] == '8'
    assert env['TOTAL_UPDATES'] == ('100' if arm == 'e0' else '30')
    assert settings['data.train_batch_size'] == '8'
    assert settings['actor_rollout_ref.rollout.n'] == '8'
    assert settings['algorithm.adv_estimator'] == ('grpo' if arm in ('e0', 'e1') else 'tau_gigpo')
    assert snapshot['experiment']['dynamic_filter']['enable'] == (arm in ('e1', 'e3'))
    assert settings['trainer.save_freq'] == settings['trainer.test_freq'] == '10'
    assert settings['trainer.resume_mode'] == 'disable'
    assert settings['trainer.logger'] == '[console,swanlab]'
    assert settings['ray_kwargs.ray_init.runtime_env.env_vars.TAU3_SWANLAB_CONTINUITY'] == "'1'"
    assert settings['ray_kwargs.ray_init.runtime_env.env_vars.TAU3_KEEP_COMPLETE_BOUNDARY'] == "'1'"
    assert settings['trainer.max_actor_ckpt_to_keep'] == '1'
    assert settings['actor_rollout_ref.actor.checkpoint.save_contents'] == '[model,optimizer,extra]'
    assert settings['actor_rollout_ref.rollout.val_kwargs.n'] == '4'
    assert settings['actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu'] == '1'
    for role in ('actor', 'ref'):
        assert settings[f'actor_rollout_ref.{role}.fsdp_config.dtype'] == 'float32'
        assert settings[f'actor_rollout_ref.{role}.fsdp_config.model_dtype'] == 'float32'
    assert settings['actor_rollout_ref.actor.fsdp_config.param_offload'] == 'false'
    assert settings['actor_rollout_ref.actor.fsdp_config.optimizer_offload'] == 'false'
    assert settings['actor_rollout_ref.ref.fsdp_config.param_offload'] == 'true'
    assert settings['actor_rollout_ref.model.use_remove_padding'] == 'false'
    assert settings['actor_rollout_ref.model.use_fused_kernels'] == 'false'
    assert settings['algorithm.rollout_correction.bypass_mode'] == 'false'
    for key, value in {'VERL_QWEN35_FLA_IEEE': '1',
                       'VERL_QWEN35_TRIM_PADDING': 'experimental_both',
                       'TRITON_F32_DEFAULT': 'ieee'}.items():
        assert env[key] == value
        assert settings[f'ray_kwargs.ray_init.runtime_env.env_vars.{key}'].strip("'") == value
    assert settings['actor_rollout_ref.rollout.multi_turn.tool_execution_mode'] == 'sequential'
    assert settings['data.apply_chat_template_kwargs.enable_thinking'] == 'false'
    assert settings['ray_kwargs.ray_init.runtime_env.env_vars.TAU3_GRPO_ARM'] == f"'{arm}'"
    assert ('ray_kwargs.ray_init.runtime_env.env_vars.TAU3_E0_DISCOVERY_SECONDS' in settings) == (arm == 'e0')
    assert env['MODEL_PATH'] == str(tmp_path / 'checkpoints/sft-merged/new-off')
    assert env['RESULTS_DIR'].endswith(f'/{arm}_seed42')
    assert not (tmp_path / 'runs').exists()

    checkpoint = Path(env['RESULTS_DIR']) / 'global_step_30'
    command, resumed, _ = run.stage_config(arm, 50, resume_from=checkpoint)
    output = subprocess.check_output(command, env=dict(resumed, TAU3_DRY_RUN='1'), text=True)
    args = shlex.split(output.splitlines()[-1])
    settings = dict(arg.lstrip('+').split('=', 1) for arg in args if '=' in arg)
    assert settings['trainer.resume_mode'] == 'resume_path'
    assert settings['trainer.resume_from_path'] == str(checkpoint)
    assert settings['trainer.total_training_steps'] == '50'
    assert 'TAU3_E0_DISCOVERY_SECONDS' not in resumed
    assert resumed['TRAIN_PARQUET'].endswith('train_schedule.parquet')

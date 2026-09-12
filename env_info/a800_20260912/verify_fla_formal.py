"""Verify formal profiles use the exact online-validated FLA compute path.

Run after launch_matched50.sh dry-run. Does not launch a model or GPU job.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import yaml

from tau3_grpo.paths import CODE_ROOT


def lookup(config, path):
    for key in path.split('.'):
        config = config.get(key) if config is not None else None
    return config


def main():
    root = Path(os.environ['TAU3_RUN_ROOT'])
    preflight = root / 'formal50-preflight-20260912'
    # The earlier smoke captured launcher preflight stdout before Hydra YAML.
    baseline_lines = (root / 'fla-online-20260912/smoke.hydra.yaml').read_text().splitlines()
    start = next(i for i, line in enumerate(baseline_lines) if line.startswith('model_engine:'))
    baseline = yaml.safe_load('\n'.join(baseline_lines[start:]))
    evidence = CODE_ROOT / 'env_info/a800_20260912/fla_adoption_evidence'
    source = json.loads((evidence / 'source-sha256.json').read_text())['files']
    hashes = {}
    for name in ('verl/verl/utils/qwen35_fla_ieee.py',
                 'verl/verl/workers/fsdp_workers.py',
                 'verl/verl/models/transformers/monkey_patch.py'):
        hashes[name] = hashlib.sha256((CODE_ROOT / name).read_bytes()).hexdigest()
        assert hashes[name] == source[name], name
    keys = ['actor_rollout_ref.model.use_remove_padding',
            'actor_rollout_ref.model.use_fused_kernels',
            'actor_rollout_ref.model.override_config.attn_implementation',
            'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu',
            'actor_rollout_ref.actor.ppo_mini_batch_size',
            'actor_rollout_ref.actor.optim.lr',
            'actor_rollout_ref.actor.kl_loss_coef',
            'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu',
            'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu',
            'actor_rollout_ref.rollout.max_model_len',
            'actor_rollout_ref.rollout.max_num_seqs',
            'actor_rollout_ref.rollout.gpu_memory_utilization',
            'actor_rollout_ref.rollout.n', 'data.train_batch_size',
            'algorithm.rollout_correction.bypass_mode']
    keys += [f'actor_rollout_ref.{role}.fsdp_config.{field}'
             for role in ('actor', 'ref')
             for field in ('dtype', 'model_dtype', 'mixed_precision', 'param_offload', 'optimizer_offload')]
    keys += [f'ray_kwargs.ray_init.runtime_env.env_vars.{name}' for name in (
        'VERL_QWEN35_FLA_IEEE', 'VERL_QWEN35_TRIM_PADDING', 'TRITON_F32_DEFAULT',
        'VERL_QWEN35_COMPACT_HEAD', 'VERL_QWEN35_COMPACT_CHUNK',
        'VERL_QWEN35_COMPACT_BACKEND', 'VERL_QWEN35_FIX_PADDING')]
    profiles = {}
    for arm in ('e0', 'e1', 'e2', 'e3'):
        config = yaml.safe_load((preflight / f'{arm}.matched.hydra.yaml').read_text())
        values = {key: lookup(config, key) for key in keys}
        assert all(value == lookup(baseline, key) for key, value in values.items()), values
        assert lookup(config, 'trainer.save_freq') == lookup(config, 'trainer.test_freq') == 10
        assert lookup(config, 'actor_rollout_ref.rollout.val_kwargs.n') == 4
        assert lookup(config, 'trainer.logger') == ['console', 'swanlab']
        assert lookup(config, 'trainer.max_actor_ckpt_to_keep') == 1
        assert lookup(config, 'actor_rollout_ref.actor.checkpoint.save_contents') == ['model', 'optimizer', 'extra']
        profiles[arm] = values
    import torch
    import fla
    from transformers.integrations import sdpa_attention
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
    from verl.utils import qwen35_fla_ieee
    from verl.trainer.ppo import ray_trainer
    assert Path(qwen35_fla_ieee.__file__).resolve() == CODE_ROOT / 'verl/verl/utils/qwen35_fla_ieee.py'
    assert Path(ray_trainer.__file__).resolve() == CODE_ROOT / 'verl/verl/trainer/ppo/ray_trainer.py'
    assert Path(fla.__file__).resolve().is_relative_to('/root/autodl-tmp/tau3-perf-20260912/overlay')
    assert importlib.metadata.version('flash-linear-attention') == '0.5.2'
    qwen35_fla_ieee.prepare_fla_ieee_runtime()
    assert hf.FusedRMSNormGated is None
    assert os.environ['TRITON_F32_DEFAULT'] == 'ieee'
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not sdpa_attention.use_gqa_in_sdpa(None, torch.empty(1, dtype=torch.float32))
    assert 'fa2-overlay' not in os.environ.get('PYTHONPATH', '')
    manifest = CODE_ROOT / 'results/analysis/rl_curriculum50_20260912/manifests/areal_airline_train_seed42.jsonl'
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert manifest_sha == '641bde73c1495c59b5c0a87cfc84b9e00c0b5ffd2f86d10fd2205aaf2143adae'
    report = {'passed': True, 'validation_kind': 'source_identity_runtime_and_formal_config_no_model_run',
              'source_hashes': hashes, 'manifest_sha256': manifest_sha,
              'fla_path': fla.__file__, 'fla_version': '0.5.2', 'profiles': profiles}
    (preflight / 'fla-adoption.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'passed': True, 'four_profiles_match_validated_compute_path': True,
                      'manifest_sha256': manifest_sha, 'fla_version': '0.5.2'}))


if __name__ == '__main__':
    main()

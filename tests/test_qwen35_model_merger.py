"""Actual Transformers serialization regressions for native FSDP exports."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_qwen35 import tiny_model
from transformers import AutoModelForImageTextToText
from verl.model_merger import base_model_merger
from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger
from verl.utils.qwen35_checkpoint import qwen35_tensor_manifest, verify_qwen35_export


@pytest.mark.parametrize('tied', [False, True])
def test_fsdp_export_roundtrip_preserves_every_native_tensor(tmp_path, monkeypatch, tied):
    torch.manual_seed(42)
    model = tiny_model(tied).to(torch.bfloat16)
    model.config.architectures = [type(model).__name__]
    config_dir = tmp_path / 'config'
    model.config.save_pretrained(config_dir)
    source = {key: value.clone() for key, value in model.state_dict().items()}
    monkeypatch.setattr(base_model_merger, 'hf_processor', lambda *a, **k: None)
    monkeypatch.setattr(base_model_merger, 'hf_tokenizer', lambda *a, **k: None)
    destination = tmp_path / 'export'
    merger = FSDPModelMerger(ModelMergerConfig(operation='merge', backend='fsdp',
        target_dir=str(destination), hf_model_config_path=str(config_dir)))
    merger.save_hf_model_and_tokenizer(source)
    manifest = qwen35_tensor_manifest(destination)
    assert 'model.language_model.layers.0.input_layernorm.weight' in manifest
    receipt = json.loads((destination / 'export-verification.json').read_text())
    assert receipt['native_format'] and receipt['verified_tensors'] == len(manifest)
    restored, info = AutoModelForImageTextToText.from_pretrained(
        destination, dtype=torch.bfloat16, output_loading_info=True)
    assert not info['missing_keys'] and not info['unexpected_keys']
    assert all(torch.equal(value, restored.state_dict()[key]) for key, value in source.items())


def test_reverse_conversion_bug_is_rejected_before_gpu_start(tmp_path):
    model = tiny_model(True).to(torch.bfloat16)
    # Layout observed in the failed production exports, independent of the
    # local Transformers version (newer versions may fix reverse conversion).
    saved = {k.replace('model.', 'model.language_model.language_model.', 1): v.clone()
             for k, v in model.state_dict().items()}
    save_file(saved, tmp_path / 'model.safetensors')
    with pytest.raises(ValueError, match='Non-native'):
        qwen35_tensor_manifest(tmp_path)


def test_export_verification_detects_changed_tensor_values(tmp_path):
    model = tiny_model(False).to(torch.bfloat16)
    source = {k: v.clone() for k, v in model.state_dict().items()}
    model.save_pretrained(tmp_path, state_dict=dict(source), save_original_format=False)
    shard = tmp_path / 'model.safetensors'
    saved = load_file(shard)
    saved['model.language_model.norm.weight'].add_(1)
    save_file(saved, shard)
    with pytest.raises(ValueError, match='differs from reconstructed'):
        verify_qwen35_export(tmp_path, source, tied=False)


def test_export_verification_rejects_missing_untied_head(tmp_path):
    model = tiny_model(False).to(torch.bfloat16)
    source = {k: v.clone() for k, v in model.state_dict().items()}
    model.save_pretrained(tmp_path, state_dict=dict(source), save_original_format=False)
    shard = tmp_path / 'model.safetensors'
    saved = load_file(shard)
    saved.pop('lm_head.weight')
    save_file(saved, shard)
    with pytest.raises(ValueError, match='lost FSDP tensors'):
        verify_qwen35_export(tmp_path, source, tied=False)

"""Evaluate an SFT model with the unchanged answer-only validation target."""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import yaml

from tau3_grpo.paths import resolve_project_path
from tau3_grpo.training.sft.dataset import TrajectorySFTDataset, collate_fn_padding


def validation_dataset(config, tokenizer):
    """Ignore training thinking flags deliberately for comparable answer loss."""
    source = resolve_project_path(config['data']['validation_jsonl'])
    tools_path = resolve_project_path(config['data']['tool_config'])
    tools = [item['tool_schema'] for item in yaml.safe_load(tools_path.read_text())['tools']]
    dataset = TrajectorySFTDataset(source, tokenizer, tools=tools,
        max_length=int(config['data']['max_length']), expected_size=int(config['data'].get('expected_validation_size', 5)),
        enable_thinking=False, supervise_reasoning=False,
        preserve_historical_reasoning=False)
    digest = hashlib.sha256(json.dumps(dataset.examples, sort_keys=True).encode()).hexdigest()
    return dataset, {'validation_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                     'rendered_examples_sha256': digest,
                     'tool_config_sha256': hashlib.sha256(tools_path.read_bytes()).hexdigest()}


def baseline_loss(summary, stats):
    if summary.get('enable_thinking') or summary.get('supervise_reasoning'):
        raise ValueError('Baseline must report answer-only validation loss')
    if summary.get('validation') != stats:
        raise ValueError('Baseline validation token statistics differ')
    return float(summary['validation_metrics']['eval_loss'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='Full model or LoRA adapter directory')
    parser.add_argument('--base-model', type=Path, help='Exact original base; required for a LoRA adapter')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-summary', type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Choose a new output path; evaluation reports are not overwritten')
    config_path = resolve_project_path(str(args.config))
    config = yaml.safe_load(config_path.read_text())
    model_path = args.model.expanduser().resolve()
    is_adapter = (model_path / 'adapter_config.json').is_file()
    if is_adapter and args.base_model is None:
        parser.error('--base-model is required for a LoRA adapter')
    import torch
    from transformers import AutoTokenizer, TrainingArguments
    from tau3_grpo.models.compat import load_policy_model, model_family, require_training_runtime
    from tau3_grpo.training.sft.trainer import Qwen35SFTTrainer
    base = args.base_model.expanduser().resolve() if is_adapter else model_path
    if model_family(str(base)) != 'qwen35':
        raise ValueError('This evaluator currently supports Qwen3.5 only')
    require_training_runtime(torch, 'qwen35')
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    dataset, provenance = validation_dataset(config, tokenizer)
    reference = None
    if args.baseline_summary:
        reference = baseline_loss(json.loads(args.baseline_summary.read_text()), dataset.token_stats())
    model = load_policy_model(str(base), torch_dtype=torch.bfloat16 if is_adapter else torch.float32,
        attn_implementation=config['model'].get('attn_implementation', 'sdpa'), trust_remote_code=False)
    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path, is_trainable=False)
    training_args = TrainingArguments(output_dir=str(args.output.parent / 'eval-runtime'),
        per_device_eval_batch_size=1, bf16=True, report_to='none', remove_unused_columns=False)
    trainer = Qwen35SFTTrainer(model=model, args=training_args, eval_dataset=dataset,
        data_collator=partial(collate_fn_padding, pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer)
    metrics = trainer.evaluate()
    result = {'target': 'answer_only', 'model': str(model_path), 'base_model': str(base),
              'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'validation': dataset.token_stats(), 'metrics': metrics, **provenance,
              'not_task_success_evaluation': True}
    if reference is not None:
        result.update(baseline_answer_only_loss=reference,
                      loss_difference=metrics['eval_loss'] - reference,
                      baseline_provenance_note='Legacy summary token statistics checked; source hash may be unavailable')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

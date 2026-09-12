"""Audit rollout/HF token scores on identical generated tokens, without RL."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoModelForImageTextToText
from tau3_grpo.paths import CODE_ROOT

sys.path.insert(0, str(CODE_ROOT/'tests'))
from test_qwen35_compact_head import actor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True)
    args = p.parse_args()
    samples = json.loads(args.samples.read_text())
    model = AutoModelForImageTextToText.from_pretrained(args.model,
        dtype=torch.float32, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    if os.environ.get('VERL_QWEN35_FIX_PADDING','0') == '1':
        from verl.utils.qwen35_padding import install_qwen35_padding_guard
        install_qwen35_padding_guard(model)
    a = actor(model)
    rows = []
    for i, sample in enumerate(samples):
        prompt, response = sample['prompt_token_ids'], sample['response_token_ids']
        rollout = torch.tensor(sample['rollout_log_probs'], device='cuda')
        if not response:
            continue
        outputs = {}
        for mode in ('unpadded', 'production_left_padding'):
            pad = max(0, 8192-len(prompt)) if mode == 'production_left_padding' else 0
            # Use tokenizer's actual padding token, not an assumed ID.
            pad_id = model.config.text_config.pad_token_id
            if pad_id is None:
                from transformers import AutoTokenizer
                pad_id = AutoTokenizer.from_pretrained(args.model, local_files_only=True).pad_token_id
            ids = torch.tensor([[pad_id]*pad+prompt+response], device='cuda')
            mask = torch.ones_like(ids)
            mask[:,:pad] = 0
            data = {'input_ids': ids, 'responses': ids[:,-len(response):],
                'attention_mask': mask, 'position_ids': (mask.cumsum(-1)-1).clamp_min(0),
                'response_mask': torch.ones((1,len(response)),device='cuda',dtype=torch.long)}
            with torch.no_grad():
                logp = a._forward_micro_batch(data, 1.)['log_probs'][0]
            outputs[mode] = logp
            difference = logp-rollout
            rows.append({'sample':i, 'mode':mode, 'prompt_tokens':len(prompt),
                'response_tokens':len(response), 'left_pad_tokens':pad,
                'mean_abs_delta':float(difference.abs().mean()),
                'max_abs_delta':float(difference.abs().max()),
                'fraction_abs_delta_gt_0_1':float((difference.abs()>.1).float().mean())})
        print(json.dumps(rows[-2:]),flush=True)
    args.output.write_text(json.dumps({'kind':'token_score_audit_not_bypass_training',
        'padding_guard':os.environ.get('VERL_QWEN35_FIX_PADDING','0')=='1',
        'samples':str(args.samples), 'results':rows,
        'scope':'Native HF SDPA and GDN, FP32 master/BF16 computation; same sampled tokens at temperature 1. '
                'No update, off-policy correction, simulator, or success-rate validation.'},indent=2))


if __name__ == '__main__':
    main()

"""Small synthetic GDN precision diagnosis; no model or RL training."""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm, torch_chunk_gated_delta_rule
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def main(output):
    torch.manual_seed(42)
    device = 'cuda'
    q, k, v = [torch.randn(1, 256, 2, 128, device=device, dtype=torch.bfloat16) for _ in range(3)]
    beta = torch.randn(1, 256, 2, device=device, dtype=torch.bfloat16).sigmoid()
    g = -F.softplus(torch.randn(1, 256, 2, device=device))
    weights = torch.randn_like(v)
    results = {}
    for mode in ('native_autocast', 'native_fp32', 'fla_fp32', 'fla_bf16'):
        inputs = [x.detach().clone().requires_grad_() for x in (q, k, v, beta, g)]
        query, key, value, b, gate = inputs
        start = time.monotonic()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if mode.startswith('native'):
                with torch.autocast('cuda', enabled=mode == 'native_autocast', dtype=torch.bfloat16):
                    out, _ = torch_chunk_gated_delta_rule(query, key, value, gate, b,
                        use_qk_l2norm_in_kernel=True)
            else:
                query, key = l2norm(query), l2norm(key)
                if mode == 'fla_fp32':
                    query, key, value, b = [x.float() for x in (query, key, value, b)]
                out, _ = chunk_gated_delta_rule(query, key, value, g=gate, beta=b,
                    use_qk_l2norm_in_kernel=False, output_final_state=False)
                out = out.to(torch.bfloat16)
            loss = (out.float() * weights.float()).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        results[mode] = {'output': out.detach().float().cpu(),
                         'grad': torch.cat([x.grad.detach().float().flatten().cpu() for x in inputs]),
                         'seconds': time.monotonic() - start}
    report = {'kind': 'synthetic_gdn_precision_diagnosis_not_full_model_validation', 'pairs': []}
    for left, right in [('native_autocast', 'native_fp32'), ('native_fp32', 'fla_fp32'), ('native_fp32', 'fla_bf16')]:
        a, b = results[left], results[right]
        report['pairs'].append({'reference': left, 'candidate': right,
            'output_max_abs': float((a['output'] - b['output']).abs().max()),
            'gradient_relative_l2': float((a['grad'] - b['grad']).norm() / a['grad'].norm()),
            'gradient_cosine': float(F.cosine_similarity(a['grad'], b['grad'], dim=0))})
    report['cold_forward_backward_seconds'] = {k: v['seconds'] for k, v in results.items()}
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)

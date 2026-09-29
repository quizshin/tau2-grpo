"""Check the actual online hook against the completed 64-trajectory replay."""
import json
import os
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.qwen35_fla_ieee import prepare_fla_ieee_runtime
from diagnose_gdn_model import actor

root=Path('/root/autodl-fs/tau3-core/code/results/legacy/gdn-20260912')
os.environ.update(VERL_QWEN35_COMPACT_HEAD='1',VERL_QWEN35_COMPACT_CHUNK='256',
    VERL_QWEN35_COMPACT_BACKEND='checkpoint',VERL_QWEN35_FIX_PADDING='1',
    VERL_QWEN35_TRIM_PADDING='experimental_both',VERL_QWEN35_FLA_IEEE='1')
torch.set_num_threads(4);torch.manual_seed(42);prepare_fla_ieee_runtime()
model=AutoModelForImageTextToText.from_pretrained('/root/autodl-fs/tau3-core/code/checkpoints/sft-merged/new-off',
    dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).cuda().eval()
apply_monkey_patch(model,use_remove_padding=False,use_fused_kernels=False)
data=DataProto.load_from_disk('/root/autodl-fs/tau3-core/code/results/runs/curriculum-speed-20260912/online-smoke/update-batches/update_000001.pkl').batch
runner=actor(model);runner.param_dtype=torch.float32
report={'kind':'online_FSDP_model_hook_bridge_against_validated_replay','cases':[]}
for row in [16,56]:
    part={k:data[k][row:row+1].cuda() for k in ['input_ids','attention_mask','position_ids','responses','response_mask']}
    with torch.no_grad():actual=runner._forward_micro_batch(part,1.)['log_probs'].cpu()[0]
    expected=torch.load(root/f'rl64-ieee-replay/scores-rank{row//16}.pt',map_location='cpu',weights_only=True)['fla_trim']['log_probs'][row%16]
    mask=data['response_mask'][row].bool();delta=(actual[mask]-expected[mask]).abs()
    case={'row':row,'tokens':int(mask.sum()),'max_abs_logp':float(delta.max()),'mae_logp':float(delta.mean())}
    report['cases'].append(case);print(json.dumps(case),flush=True)
    assert torch.isfinite(actual).all() and delta.max()<1e-3
report['kernel_calls']=model._tau3_fla_ieee_calls['count'];report['passed']=True
(root/'online-fla-bridge.json').write_text(json.dumps(report,indent=2)+'\n')

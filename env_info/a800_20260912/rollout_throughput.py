"""One-engine sampling benchmark using 32 real assistant-turn prefixes.

Measures model generation only, not tools/user simulation or end-to-end RL.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from verl import DataProto
from vllm import LLM, SamplingParams

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--max-seqs',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='/root/autodl-fs/tau3-core/code/checkpoints/sft-merged/new-off')
    a=p.parse_args()
    root=Path('/root/autodl-fs/tau3-core/code/results/runs/rl-formal-full-20260912/acceptance/update-batches')
    prefixes=[]
    for path in sorted(root.glob('update_*.pkl')):
        batch=DataProto.load_from_disk(str(path)).batch
        prompt_length=batch['input_ids'].shape[-1]-batch['responses'].shape[-1]
        for row in range(len(batch)):
            mask=batch['response_mask'][row].bool()
            for i in range(len(mask)):
                if mask[i] and (i==0 or not mask[i-1]):
                    end=prompt_length+i
                    ids=batch['input_ids'][row,:end][batch['attention_mask'][row,:end].bool()].tolist()
                    if len(ids)+1024<=24576:
                        prefixes.append(ids)
    assert len(prefixes)>=32
    # Cover early and late turns instead of selecting only short initial prompts.
    prefixes=sorted(prefixes,key=len)
    prefixes=[prefixes[i*(len(prefixes)-1)//31] for i in range(32)]
    fingerprint=hashlib.sha256(json.dumps(prefixes).encode()).hexdigest()
    requests=[{'prompt_token_ids':ids} for ids in prefixes]
    llm=LLM(model=a.model,dtype='bfloat16',tensor_parallel_size=1,
        max_num_seqs=a.max_seqs,max_model_len=24576,max_num_batched_tokens=24576,
        gpu_memory_utilization=.30,enforce_eager=True,enable_prefix_caching=False,
        language_model_only=True,disable_log_stats=True)
    llm.generate(requests[:2],[SamplingParams(temperature=1.,top_p=1.,top_k=-1,
        max_tokens=32,seed=42+i) for i in range(2)],use_tqdm=False)
    results=[]
    for logprobs in (None,0):
        sampling=[SamplingParams(temperature=1.,top_p=1.,top_k=-1,max_tokens=1024,
            seed=42+i,logprobs=logprobs) for i in range(32)]
        started=time.perf_counter()
        outputs=llm.generate(requests,sampling,use_tqdm=False)
        seconds=time.perf_counter()-started
        tokens=sum(len(out.outputs[0].token_ids) for out in outputs)
        results.append({'logprobs':logprobs,'seconds':seconds,'generated_tokens':tokens,
            'tokens_per_second':tokens/seconds,'requests_per_second':32/seconds,
            'length_stops':sum(out.outputs[0].finish_reason=='length' for out in outputs)})
        print(json.dumps(results[-1]),flush=True)
        if logprobs is not None:
            scored=[]
            for ids,out in zip(prefixes[:8],outputs[:8]):
                completion=out.outputs[0]
                scored.append({'prompt_token_ids':ids,'response_token_ids':completion.token_ids,
                    'rollout_log_probs':[lp[token].logprob for token,lp in
                        zip(completion.token_ids,completion.logprobs)]})
            a.output.with_suffix('.samples.json').write_text(json.dumps(scored))
    report={'kind':'one_engine_generation_only','max_num_seqs':a.max_seqs,
        'prompt_sha256':fingerprint,'requests':32,'prompt_lengths':[len(x) for x in prefixes],
        'gpu_memory_utilization':.30,'results':results,
        'scope':'No environment or simulator; throughput does not prove full RL wall time.'}
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()

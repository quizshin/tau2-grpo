"""Bounded isolated vLLM LoRA eager/graph replay, never an online-training switch."""

import argparse
import gc
import json
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--stream", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--mode", choices=["prepare", "eager", "graph"], required=True)
    p.add_argument("--actor-pid", type=int, required=True)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    import psutil

    def available():
        name = psutil.Process(args.actor_pid).name()
        if not any(s in name for s in ("compute_log_prob", "compute_ref_log_prob", "update_actor")):
            raise RuntimeError("Production may need simulator; abort probe: " + name)

    if args.mode == "prepare":
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText

        torch.manual_seed(42)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, local_files_only=True
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                target_modules=[
                    "down_proj",
                    "gate_proj",
                    "in_proj_a",
                    "in_proj_b",
                    "in_proj_qkv",
                    "in_proj_z",
                    "k_proj",
                    "o_proj",
                    "out_proj",
                    "q_proj",
                    "up_proj",
                    "v_proj",
                ],
                bias="none",
            ),
        )
        assert not (args.work / "adapter0").exists()
        model.save_pretrained(args.work / "adapter0")
        with torch.no_grad():
            for name, value in model.named_parameters():
                if "lora_B" in name:
                    value.normal_(mean=0, std=0.001)
        model.save_pretrained(args.work / "adapter1")
        (args.work / "identity.json").write_text(
            json.dumps(
                dict(
                    model=str(args.model),
                    scope="Synthetic LoRA rank16 adapters on exact repair72; not trained weights",
                    seed=42,
                    adapter0="B=0",
                    adapter1="B normal std0.001",
                ),
                indent=2,
            )
        )
        return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    records = [json.loads(p.read_text()) for p in sorted(args.stream.glob("*.json"))[:16]]
    if len(records) != 16:
        raise ValueError("Need sixteen actual rollout prefixes")
    prompts = [{"prompt_token_ids": d["output"]["prompt_ids"]} for d in records]
    available()
    start = time.monotonic()
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=24576,
        max_num_batched_tokens=24576,
        max_num_seqs=16,
        gpu_memory_utilization=0.24,
        enforce_eager=args.mode == "eager",
        enable_lora=True,
        max_lora_rank=16,
        max_loras=1,
        enable_prefix_caching=False,
        language_model_only=True,
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16]},
        disable_log_stats=False,
    )
    report = dict(
        mode=args.mode,
        engine_start_seconds=time.monotonic() - start,
        scope="16 real initial prefixes; model-only generation, not complete evaluation",
        gpu_memory_fraction=0.24,
        results=[],
    )
    path = args.work / (args.mode + ".json")

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(path)

    save()
    for idx in (0, 1):
        available()
        request = LoRARequest("probe" + str(idx), idx + 1, str(args.work / ("adapter" + str(idx))))
        params = SamplingParams(temperature=0, max_tokens=64, logprobs=0, seed=42)
        llm.generate(prompts, params, lora_request=request, use_tqdm=False)
        for repeat in range(2):
            available()
            start = time.monotonic()
            outputs = llm.generate(prompts, params, lora_request=request, use_tqdm=False)
            seconds = time.monotonic() - start
            rows = []
            for item in outputs:
                out = item.outputs[0]
                rows.append(
                    dict(
                        tokens=list(out.token_ids),
                        logprobs=[
                            float(lp[token].logprob)
                            for token, lp in zip(out.token_ids, out.logprobs)
                        ],
                    )
                )
            report["results"].append(
                dict(
                    adapter=idx,
                    repeat=repeat,
                    seconds=seconds,
                    generated_tokens=sum(len(x["tokens"]) for x in rows),
                    rows=rows,
                )
            )
            save()
    report["completed"] = True
    save()
    del llm
    gc.collect()


if __name__ == "__main__":
    main()

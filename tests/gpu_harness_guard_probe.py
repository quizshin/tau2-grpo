"""Explicit single-GPU engineering probe; never collected by default pytest.

Run only with user authorization. Constrained repetition is a cancellation
fixture, not a measurement of the model's spontaneous repetition or quality.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace


class EngineProbe:
    def __init__(self, engine, directory, max_requests):
        self.engine, self.directory, self.max_requests = engine, directory, max_requests
        self.records, self.aborts = {}, []
        self.cache_resets = 0

    def __getattr__(self, name):
        return getattr(self.engine, name)

    async def generate(self, **kwargs):
        request_id = kwargs["request_id"]
        if len(self.records) >= self.max_requests or request_id in self.records:
            raise RuntimeError("GPU probe request budget or unique identity violated")
        params = kwargs["sampling_params"]
        record = {"request_id": request_id, "started": time.monotonic(), "chunks": 0,
                  "max_tokens": params.max_tokens, "repetition_penalty": params.repetition_penalty,
                  "output_kind": str(params.output_kind), "allowed_token_ids": params.allowed_token_ids,
                  "input_tokens": len(kwargs["prompt"]["prompt_token_ids"])}
        self.records[request_id] = record
        async for output in self.engine.generate(**kwargs):
            completion = output.outputs[0]
            ids = list(completion.token_ids)
            record["chunks"] += 1
            if ids:
                record["last_nonempty_ids"] = ids
                record["last_nonempty_logprobs"] = [p[t].logprob for t, p in zip(ids, completion.logprobs, strict=True)]
            if completion.finish_reason is not None:
                record.update(finish_reason=completion.finish_reason, finished=time.monotonic(),
                              terminal_ids=ids)
            yield output

    async def abort(self, request_id):
        self.aborts.append({"request_id": request_id, "time": time.monotonic(),
                            "other_request_prefixes": {
                                key: len(value.get("last_nonempty_ids", []))
                                for key, value in self.records.items()
                                if key != request_id and "finished" not in value}})
        return await self.engine.abort(request_id)

    async def reset_prefix_cache(self, *args, **kwargs):
        self.cache_resets += 1
        return await self.engine.reset_prefix_cache(*args, **kwargs)

    def save(self):
        (self.directory / "engine-receipts.json").write_text(json.dumps({
            "requests": self.records, "abort_calls": self.aborts, "cache_resets": self.cache_resets},
            indent=2, allow_nan=False) + "\n")


async def exercise(args, report):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from verl.workers.config import RolloutConfig
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from tau3_grpo.models.generation_guard import GenerationGuard
    from tau3_grpo.training.rl.update_guard import HarnessUpdateHalt, enforce_update_guard
    from verl import DataProto

    assert torch.cuda.device_count() == 1, "This probe requires exactly one visible GPU"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    engine_options = dict(model=str(args.model), dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=24576, max_num_seqs=4, max_num_batched_tokens=4096,
        gpu_memory_utilization=0.65, enforce_eager=True, enable_prefix_caching=True,
        enable_chunked_prefill=True, trust_remote_code=False, seed=42,
        limit_mm_per_prompt={"image": 0, "video": 0}, generation_config="vllm",
        logprobs_mode="processed_logprobs")
    report.update(engine_options=engine_options, gpu=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__, tokenizer_config_sha256=hashlib.sha256(
                      (args.model / "tokenizer_config.json").read_bytes()).hexdigest())
    print("INITIALIZING_ENGINE", flush=True)
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_options))
    probe = EngineProbe(engine, args.output, args.max_requests)

    def server(mode):
        # Exercise the production generate/abort methods against a real engine;
        # bypass Ray service allocation, not the generation or cancellation code.
        obj = vLLMHttpServer.__new__(vLLMHttpServer)
        obj.engine = probe
        obj.config = RolloutConfig(name="vllm", prompt_length=8192, response_length=16384,
            max_model_len=24576, generation_guard=asdict(GenerationGuard(mode=mode)),
            calculate_log_probs=True)
        obj.model_config = SimpleNamespace(tokenizer=tokenizer, processor=None, lora_rank=0, lora={})
        obj.global_steps, obj.replica_rank = 0, 0
        return obj

    servers = {mode: server(mode) for mode in ("off", "observe", "abort")}
    params = dict(temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0, logprobs=True, seed=42)
    token = tokenizer.encode(" x", add_special_tokens=False)
    assert len(token) == 1 and token[0] not in tokenizer.all_special_ids

    def prompt(text):
        return tokenizer.apply_chat_template([{"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)

    async def generate(name, mode, text, **overrides):
        output = await asyncio.wait_for(servers[mode].generate(prompt_ids=prompt(text),
            sampling_params={**params, "max_tokens": 1024, **overrides}, request_id=name), timeout=180)
        record = probe.records[name]
        assert output.token_ids == record["last_nonempty_ids"]
        assert output.log_probs == record["last_nonempty_logprobs"]
        assert len(output.token_ids) == len(output.log_probs) and all(map(math.isfinite, output.log_probs))
        item = {"name": name, "tokens": len(output.token_ids), **output.extra_fields,
                "text": tokenizer.decode(output.token_ids, skip_special_tokens=True)}
        report["cases"].append(item)
        print(json.dumps({"case": name, "tokens": len(output.token_ids),
                          "finish": output.extra_fields["finish_reason"]}), flush=True)
        return output

    try:
        if args.concurrency_control:
            long_text = "Write a detailed essay about the history of navigation, with varied examples and no lists."
            repeated = "Repeat x, separated by spaces."
            outputs = {}
            for name in ("serial-a", "serial-b"):
                outputs[name] = await generate(name, "off", long_text, min_tokens=512, max_tokens=512)
            for label, repeat_mode, normal_mode in (("no-abort", "off", "off"),
                                                    ("abort-a", "abort", "abort"),
                                                    ("abort-b", "abort", "abort")):
                repeat_output, normal_output = await asyncio.gather(
                    generate(label + "-repeat", repeat_mode, repeated, allowed_token_ids=token, max_tokens=512),
                    generate(label + "-normal", normal_mode, long_text, min_tokens=512, max_tokens=512))
                assert normal_output.extra_fields["finish_reason"] == "length"
                assert not normal_output.extra_fields["generation_guard"] and len(normal_output.token_ids) == 512
                assert repeat_output.extra_fields["finish_reason"] == ("length" if repeat_mode == "off" else "abort")
                outputs[label + "-normal"] = normal_output
            outputs["serial-after"] = await generate("serial-after", "off", long_text, min_tokens=512, max_tokens=512)
            comparisons = []
            for left, right in (("serial-a", "serial-b"), ("serial-a", "no-abort-normal"),
                                ("no-abort-normal", "abort-a-normal"), ("abort-a-normal", "abort-b-normal"),
                                ("serial-a", "serial-after")):
                a, b = outputs[left].token_ids, outputs[right].token_ids
                first = next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)
                comparisons.append({"left": left, "right": right, "first_different_token_zero_based": first})
            report["concurrency_comparisons"] = comparisons
            report["active_requests_at_end"] = len(engine.output_processor.request_states)
            report["request_count"] = len(probe.records)
            assert report["active_requests_at_end"] == 0 and probe.cache_resets == 0
            return
        short = "Answer with only the capital city of France."
        if not args.processed_confirmation:
            baseline = await generate("short-off", "off", short, max_tokens=128)
            guarded = await generate("short-abort", "abort", short, max_tokens=128)
            assert baseline.token_ids == guarded.token_ids
            assert guarded.extra_fields["finish_reason"] == "stop" and not guarded.extra_fields["generation_guard"]
        await generate("penalty-1.15", "abort", short, max_tokens=128, repetition_penalty=1.15)
        assert probe.records["penalty-1.15"]["repetition_penalty"] == 1.15

        repeated = "Repeat x, separated by spaces."
        if not args.processed_confirmation:
            off = await generate("repeat-off", "off", repeated, allowed_token_ids=token)
            assert len(off.token_ids) == 1024 and off.extra_fields["finish_reason"] == "length"
        observed = await generate("repeat-observe", "observe", repeated, allowed_token_ids=token)
        stopped = await generate("repeat-abort", "abort", repeated, allowed_token_ids=token)
        assert len(observed.token_ids) == 1024 and observed.extra_fields["finish_reason"] == "length"
        assert max(map(abs, observed.log_probs)) < 1e-6  # One allowed token after logits processing.
        assert observed.extra_fields["generation_guard"] and not observed.extra_fields["generation_guard"]["cancelled"]
        assert stopped.extra_fields["finish_reason"] == "abort"
        assert 128 <= len(stopped.token_ids) < 1024 and stopped.extra_fields["generation_guard"]["cancelled"]
        edge = await generate("repeat-natural-length", "abort", repeated, allowed_token_ids=token, max_tokens=128)
        assert edge.extra_fields["finish_reason"] == "length"
        assert not edge.extra_fields["generation_guard"]["cancelled"]

        long_text = "Write a detailed essay about the history of navigation, with varied examples and no lists."
        if not args.processed_confirmation:
            long_baseline = await generate("long-off", "off", long_text, min_tokens=512, max_tokens=512)
        concurrent_repeat, concurrent_normal = await asyncio.gather(
            generate("concurrent-repeat", "abort", repeated, allowed_token_ids=token),
            generate("concurrent-normal", "abort", long_text, min_tokens=512, max_tokens=512))
        assert concurrent_repeat.extra_fields["finish_reason"] == "abort"
        assert not concurrent_normal.extra_fields["generation_guard"]
        assert len(concurrent_normal.token_ids) == 512
        if not args.processed_confirmation:
            report["concurrent_normal_matches_sequential_tokens"] = concurrent_normal.token_ids == long_baseline.token_ids
        abort_time = next(x["time"] for x in probe.aborts if x["request_id"] == "concurrent-repeat")
        assert probe.records["concurrent-normal"]["started"] < abort_time < probe.records["concurrent-normal"]["finished"]
        if not args.processed_confirmation:
            after = await generate("after-cancellation", "abort", short, max_tokens=128)
            assert after.extra_fields["finish_reason"] == "stop"
        assert probe.cache_resets == 0

        # Three actual native loops with real GPU-generated constrained prefixes.
        from tau3_grpo.data.manifest import read_manifest
        from tau3_grpo.data.parquet_builder import build_row
        from tau3_grpo.envs.registry import SESSIONS
        from tau3_grpo.evaluation.runtime import Endpoint
        from tau3_grpo.evaluation.token_runtime import make_loop
        from tau3_grpo.paths import MANIFEST_ROOT, TAU2_BENCH_ROOT

        entry = read_manifest(MANIFEST_ROOT / "areal_airline_selection_seed42.jsonl")[0]
        row = build_row(entry, policy=(TAU2_BENCH_ROOT / "data/tau2/domains/airline/policy.md").read_text(),
                        split="selection", seed=42)
        native = []
        for estimator in ("grpo", "tau_gigpo", "mt_gtpo"):
            class Manager:
                async def generate(self, *, sampling_params, **kwargs):
                    return await servers["abort"].generate(**kwargs, sampling_params={
                        **sampling_params, "temperature": 0.0, "allowed_token_ids": token, "seed": 42})

            directory = args.output / estimator
            directory.mkdir()
            loop, _ = await make_loop(tokenizer=tokenizer, manager=Manager(),
                user=Endpoint("unused", "http://127.0.0.1:1"), directory=directory,
                estimator=estimator, evaluation=False)
            output = await asyncio.wait_for(loop.run(dict(params), raw_prompt=row["prompt"],
                extra_info=row["extra_info"], tau3_sampling_identity={"seed": 42, "task_id": entry.task_id}), 180)
            facts = json.loads(output.extra_fields["trajectory_facts_json"])
            assert output.reward_score == 0 and not facts["terminal"]["scored"]
            assert facts["terminal"]["termination_detail"] == "repetition_detected"
            assert facts["turns"][-1]["parse_status"] == "not_attempted"
            assert len(output.response_ids) == len(output.response_logprobs) == len(output.response_mask)
            assert all(output.response_mask) and SESSIONS.active_count() == 0
            (directory / "facts.json").write_text(json.dumps(facts, indent=2) + "\n")
            native.append(output)
            report["native_loops"].append({"estimator": estimator, "tokens": len(output.response_ids),
                                           "termination_detail": facts["terminal"]["termination_detail"]})
            print("NATIVE_LOOP_PASSED", estimator, flush=True)

        width = max(len(x.response_ids) for x in native)
        ids = torch.zeros((3, width), dtype=torch.long, device="cuda")
        masks = torch.zeros_like(ids)
        logprobs = torch.zeros((3, width), dtype=torch.float32, device="cuda")
        for k, output in enumerate(native):
            size = len(output.response_ids)
            ids[k, :size] = torch.tensor(output.response_ids, device="cuda")
            masks[k, :size] = torch.tensor(output.response_mask, device="cuda")
            logprobs[k, :size] = torch.tensor(output.response_logprobs, device="cuda")
        batch = DataProto.from_dict(tensors={"responses": ids, "response_mask": masks,
                                             "rollout_log_probs": logprobs},
            non_tensors={"trajectory_facts_json": np.array([
                x.extra_fields["trajectory_facts_json"] for x in native], dtype=object)})
        would_update = False
        try:
            enforce_update_guard(batch, {"mode": "halt", "repetition_fraction": .0625},
                step=1, output_dir=args.output, resolved_config={"scope": "GPU engineering probe"})
            would_update = True
        except HarnessUpdateHalt:
            report["update_guard_halted"] = True
        assert not would_update and report.get("update_guard_halted")
        fault = json.loads(next(args.output.rglob("fault.json")).read_text())
        assert fault["decision"]["health"]["candidates"] == 3 and not fault["is_resumable_checkpoint"]
        report["update_guard_scope"] = "Actual generated prefixes in a GPU DataProto; no optimizer or distributed trainer initialized"
        report["active_requests_at_end"] = len(engine.output_processor.request_states)
        assert report["active_requests_at_end"] == 0
        report["request_count"] = len(probe.records)
    finally:
        probe.save()
        engine.shutdown(timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--concurrency-control", action="store_true")
    parser.add_argument("--processed-confirmation", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "cases": [], "native_loops": [], "started": time.time(),
              "scope": "Single GPU real generation/cancellation; constrained repetition fixture; no model quality claim"}
    try:
        asyncio.run(exercise(args, report))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.time() - report["started"]
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

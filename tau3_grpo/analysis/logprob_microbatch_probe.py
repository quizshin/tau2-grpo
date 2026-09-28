"""Bounded reference-forward replay; no optimizer, sampling or production writes."""

import argparse
import gc
import json
import os
import signal
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "batch", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--actor-pid", type=int, required=True)
    p.add_argument("--max-seconds", type=int, default=600)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("probe budget")))
    signal.alarm(args.max_seconds)
    import psutil
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoModelForImageTextToText
    from verl.utils.qwen35_compact_head import compact_forward
    from verl.utils.qwen35_fla_ieee import install_fla_ieee, prepare_fla_ieee_runtime
    from verl.utils.qwen35_padding import install_qwen35_padding_guard

    from verl import DataProto

    def available():
        name = psutil.Process(args.actor_pid).name()
        if not any(s in name for s in ("compute_log_prob", "compute_ref_log_prob", "update_actor")):
            raise RuntimeError("Production may need simulator again; stopping probe: " + name)

    os.environ.update(
        VERL_QWEN35_COMPACT_HEAD="1",
        VERL_QWEN35_COMPACT_BACKEND="checkpoint",
        VERL_QWEN35_TRIM_PADDING="experimental_both",
        VERL_QWEN35_FIX_PADDING="1",
    )
    torch.cuda.set_per_process_memory_fraction(0.29)
    prepare_fla_ieee_runtime()
    torch.manual_seed(42)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    )
    install_qwen35_padding_guard(model)
    install_fla_ieee(model)
    # Preload on CPU; acquire spare GPU memory only during production compute.
    waiting = time.monotonic()
    while True:
        try:
            available()
            break
        except RuntimeError:
            if time.monotonic() - waiting > 1200:
                raise TimeoutError("No simulator-idle window in20 minutes")
            time.sleep(5)
    model = model.cuda().eval()
    actor = type("ActorView", (), {})()
    actor.actor_module, actor.device_name, actor.param_dtype = model, "cuda", torch.float32
    actor.config = OmegaConf.create({"calculate_sum_pi_squared": False})
    for flag in ("use_remove_padding", "use_ulysses_sp", "use_fused_kernels", "use_prefix_grouper"):
        setattr(actor, flag, False)
    packet = DataProto.load_from_disk(args.batch)
    data = packet.batch
    order = data["attention_mask"].sum(-1).argsort().tolist()
    pairs = [order[:2], order[-2:]]
    report = dict(
        scope="Reference model unsharded forward; not actor/FSDP or full RL validation",
        memory_fraction_cap=0.29,
        input_batch=str(args.batch),
        pairs=pairs,
        logprob_max_abs_gate=1e-5,
        results=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.output)

    def forward(rows):
        available()
        micro = {
            k: data[k][rows].cuda()
            for k in ("input_ids", "responses", "response_mask", "attention_mask", "position_ids")
        }
        # Production IEEE wrapper deliberately rejects batches >1. Only this
        # isolated probe opts into the same FP32 recurrence without that guard.
        from verl.utils.qwen35_padding import fla_with_native_qk_norm

        modules = [m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]
        original = [m.chunk_gated_delta_rule for m in modules]
        try:
            if len(rows) > 1:
                for m in modules:
                    m.chunk_gated_delta_rule = fla_with_native_qk_norm
            with torch.no_grad():
                result = compact_forward(actor, micro, 1.0, False, 256)["log_probs"]
        finally:
            for m, kernel in zip(modules, original):
                m.chunk_gated_delta_rule = kernel
        return result.cpu()

    save()
    try:
        for rows in pairs:
            item = dict(rows=rows, lengths=data["attention_mask"][rows].sum(-1).tolist())
            report["results"].append(item)
            for batch_size in (1, 2):
                try:
                    chunks = [[i] for i in rows] if batch_size == 1 else [rows]
                    # Warm each exact shape before timing, preserving original tokens.
                    for chunk in chunks:
                        forward(chunk)
                    timings, peaks, values = [], [], None
                    for _ in range(2):
                        gc.collect()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.monotonic()
                        values = torch.cat([forward(chunk) for chunk in chunks])
                        torch.cuda.synchronize()
                        timings.append(time.monotonic() - start)
                        peaks.append(torch.cuda.max_memory_allocated() / 2**30)
                    item[str(batch_size)] = dict(
                        seconds=timings,
                        peak_allocated_gib=peaks,
                        finite=bool(torch.isfinite(values).all()),
                    )
                    if batch_size == 1:
                        reference = values
                    else:
                        mask = data["response_mask"][rows].bool().cpu()
                        delta = (values - reference)[mask].abs()
                        item["masked_max_abs_delta"] = float(delta.max())
                        item["numerical_pass"] = (
                            bool(torch.isfinite(values).all()) and float(delta.max()) <= 1e-5
                        )
                        item["speedup"] = sum(item["1"]["seconds"]) / sum(item["2"]["seconds"])
                except torch.cuda.OutOfMemoryError as exc:
                    item[str(batch_size)] = dict(oom=True, error=str(exc)[:300])
                    torch.cuda.empty_cache()
                save()
        report["completed"] = True
    except Exception as exc:
        report["stopped_error"] = repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()

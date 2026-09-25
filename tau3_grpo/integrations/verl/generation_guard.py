"""Consume cumulative engine output without losing a prefix on guard cancellation."""
from __future__ import annotations

import copy
import math

from tau3_grpo.models.generation_guard import detect_repetition


async def consume_guarded_generation(generator, *, tokenizer, config, cancel, require_logprobs):
    final = None
    saved = None
    evidence = None
    cancelled = False
    previous = []
    async for output in generator:
        if len(output.outputs) != 1:
            raise ValueError("Generation guard requires exactly one cumulative output")
        completion = output.outputs[0]
        ids = list(completion.token_ids)
        if not ids and cancelled and completion.finish_reason == "abort":
            # An engine abort notification can carry no tokens. Never let it erase
            # the last attested prefix or claim that the model emitted an EOS.
            final = copy.deepcopy(saved)
            final.outputs[0].finish_reason = "abort"
            if hasattr(completion, "stop_reason"):
                final.outputs[0].stop_reason = completion.stop_reason
            break
        if ids[:len(previous)] != previous:
            raise ValueError("Generation guard requires append-only cumulative token output")
        if require_logprobs:
            probs = completion.logprobs
            if probs is None or len(probs) != len(ids):
                raise ValueError("Guarded output token/logprob lengths differ")
            if any(token not in row or not math.isfinite(row[token].logprob)
                   for token, row in zip(ids, probs, strict=True)):
                raise ValueError("Guarded output lacks finite sampled-token logprobs")
        previous = ids
        final = output
        saved = copy.deepcopy(output)
        if evidence is None:
            evidence = detect_repetition(ids, tokenizer.decode(ids, skip_special_tokens=True), config)
        # Natural completion takes precedence over cancellation of a finished request.
        if evidence is not None and config.mode == "abort" and not cancelled and completion.finish_reason is None:
            result = await cancel()
            if not result.get("aborted"):
                raise RuntimeError("Repetition cancellation was not acknowledged")
            cancelled = True
    if final is None or not final.outputs[0].token_ids:
        raise ValueError("Guarded generation produced no attested tokens")
    if final.outputs[0].finish_reason not in {"stop", "length", "abort"}:
        raise ValueError("Guarded stream ended without a terminal engine receipt")
    if final.outputs[0].finish_reason == "abort" and not cancelled:
        raise RuntimeError("External generation abort is not a model repetition failure")
    if evidence is not None:
        evidence.update(cancel_requested=cancelled, final_tokens=len(final.outputs[0].token_ids),
                        cancelled=cancelled and final.outputs[0].finish_reason == "abort")
    return final, evidence

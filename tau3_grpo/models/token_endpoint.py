"""Strict vLLM completion adapter: token IDs in and actual token IDs out.

No chat re-render, text re-tokenization, retry, or fallback to inferred IDs.
"""
from __future__ import annotations

import asyncio
import json
import math
import urllib.request


class TokenEndpoint:
    def __init__(self, endpoint, *, transport=None, timeout=120):
        self.endpoint = endpoint
        self.transport = transport or self._post
        self.timeout = timeout

    def _post(self, payload):
        request = urllib.request.Request(
            self.endpoint.base_url.rstrip("/") + "/completions",
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.endpoint.api_key},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    async def generate(self, *, request_id, prompt_ids, sampling_params, image_data=None, video_data=None):
        from verl.workers.rollout.replica import TokenOutput

        if image_data or video_data:
            raise ValueError("Token endpoint is text-only")
        if not prompt_ids or any(type(t) is not int or t < 0 for t in prompt_ids):
            raise ValueError("prompt_ids must be actual integer tokens")
        params = dict(sampling_params)
        allowed = {"temperature", "top_p", "top_k", "min_p", "repetition_penalty", "seed",
                   "max_tokens", "logprobs"}
        if params.keys() - allowed:
            raise ValueError(f"Unsupported sampling options: {sorted(params.keys() - allowed)}")
        params.pop("logprobs", None)
        if type(params.get("max_tokens")) is not int or params["max_tokens"] <= 0:
            raise ValueError("Explicit positive max_tokens required")
        payload = {"model": self.endpoint.model, "prompt": list(prompt_ids), "n": 1,
                   "stream": False, "echo": False, "add_special_tokens": False,
                   "return_token_ids": True, "return_tokens_as_token_ids": True,
                   "logprobs": 1, **params}
        response = await asyncio.to_thread(self.transport, payload)
        choices = response.get("choices", [])
        if len(choices) != 1 or response.get("model") != self.endpoint.model:
            raise ValueError("Token response model/choice identity differs")
        choice = choices[0]
        if choice.get("prompt_token_ids") != list(prompt_ids):
            raise ValueError("Server did not attest the exact prompt token IDs")
        ids = choice.get("token_ids")
        if (not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids)
                or len(ids) > params["max_tokens"]):
            raise ValueError("Missing/invalid generated token IDs; text fallback is prohibited")
        finish = choice.get("finish_reason")
        if finish not in {"stop", "length"}:
            raise ValueError(f"Unresolved generation finish reason: {finish}")
        probabilities = choice.get("logprobs") or {}
        if probabilities.get("tokens") != [f"token_id:{t}" for t in ids]:
            raise ValueError("Logprob token identity differs from generated token IDs")
        logprobs = probabilities.get("token_logprobs")
        if (not isinstance(logprobs, list) or len(logprobs) != len(ids)
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in logprobs)):
            raise ValueError("Missing/nonfinite generation logprobs")
        usage = response.get("usage") or {}
        if usage.get("prompt_tokens") != len(prompt_ids) or usage.get("completion_tokens") != len(ids):
            raise ValueError("Service usage disagrees with exact token counts")
        return TokenOutput(token_ids=ids, log_probs=logprobs, stop_reason="completed", extra_fields={
            "finish_reason": finish, "native_stop_reason": choice.get("stop_reason"),
            "logprobs_mode": "server_reported_pending_live_attestation",
            "sampling_seed": params.get("seed"), "engine_seed": None,
            "sampling_parameters": {k: params.get(k) for k in (
                "temperature", "top_p", "top_k", "repetition_penalty")},
            "raw_prompt_ids_verified": True,
        })

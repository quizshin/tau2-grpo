"""Opt-in OpenAI-compatible semantic extraction, separate from rollout clients."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv

from tau3_grpo.algorithms.anchors.semantic_state import validate_packet
from tau3_grpo.utils.hashing import sha256_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class SemanticAPIError(RuntimeError):
    """Safe public error: never include credentials or provider response bodies."""


class OpenAICompatibleSemanticModel:
    simulated = False

    def __init__(self, *, base_url, model, api_key, timeout=120, max_tokens=8192,
                 temperature=0, transport=None, thinking_mode=None):
        if not api_key or not api_key.strip():
            raise ValueError('Fill TAU3_SEMANTIC_API_KEY in code/.env before running the audit')
        parsed = urlsplit(base_url)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('Semantic base URL must be HTTPS without credentials/query/fragment')
        if not model or timeout <= 0 or max_tokens <= 0:
            raise ValueError('Model, positive timeout and positive max_tokens are required')
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._api_key = api_key.strip()
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        if thinking_mode not in (None, 'enabled', 'disabled'):
            raise ValueError('thinking_mode must be enabled, disabled or null')
        self.thinking_mode = thinking_mode
        self.transport = transport
        self.attempted_calls = 0
        self.metadata = []
        self.response_packets = {}
        self.request_timings = []
        self.raw_responses = {}
        self.on_raw_response = None

    @classmethod
    def from_env(cls, config, *, transport=None):
        load_dotenv(os.environ.get('TAU3_ENV_FILE', PROJECT_ROOT / '.env'), override=False)
        return cls(base_url=os.environ.get('TAU3_SEMANTIC_BASE_URL', ''),
                   model=config.get('model') or os.environ.get('TAU3_SEMANTIC_MODEL', ''),
                   api_key=os.environ.get('TAU3_SEMANTIC_API_KEY', ''),
                   timeout=config.get('timeout_seconds', 120),
                   max_tokens=config.get('max_tokens', 8192),
                   temperature=config.get('temperature', 0), transport=transport,
                   thinking_mode=config.get('thinking_mode'))

    @property
    def provenance(self):
        return {'provider': 'openai_compatible', 'base_url': self.base_url,
                'model': self.model, 'timeout_seconds': self.timeout,
                'max_tokens': self.max_tokens, 'temperature': self.temperature,
                'thinking_mode_requested': self.thinking_mode,
                'automatic_retries': 0}

    async def extract(self, request):
        packet = await self.extract_json(request)
        try:
            validate_packet(request['visible_messages'], packet, slot_schema=request.get('slot_schema'))
        except (TypeError, KeyError, IndexError, AttributeError):
            raise SemanticAPIError('Semantic API packet has malformed field types') from None
        return packet

    async def extract_json(self, request, *, user_payload=None):
        """Transport only; callers of this method must validate their own schema."""
        if request['prefix_sha256'] != sha256_json(request['visible_messages']):
            raise ValueError('Model request prefix changed')
        payload = {'model': self.model, 'temperature': self.temperature,
                   'max_tokens': self.max_tokens, 'stream': False,
                   'messages': [{'role': 'system', 'content': request['system']},
                                {'role': 'user', 'content': json.dumps(
                                    user_payload if user_payload is not None else
                                    {k: request[k] for k in ('schema', 'prefix_sha256', 'visible_messages')},
                                    ensure_ascii=False)}]}
        # Vendor option is opt-in; provenance records a request, not a guarantee
        # that an OpenAI-compatible gateway honored the vendor parameter.
        if self.thinking_mode is not None:
            payload['thinking'] = {'type': self.thinking_mode}
        self.attempted_calls += 1
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout), httpx.AsyncClient(timeout=self.timeout, transport=self.transport,
                                         follow_redirects=False) as client:
                response = await client.post(self.base_url + '/chat/completions',
                                             headers={'Authorization': 'Bearer ' + self._api_key},
                                             json=payload)
        except httpx.HTTPError as exc:
            raise SemanticAPIError(f'Semantic API transport error or timeout ({type(exc).__name__}); no retry') from None
        except TimeoutError:
            raise SemanticAPIError('Semantic API total deadline exceeded; no retry') from None
        finally:
            self.request_timings.append({'prefix_sha256': request['prefix_sha256'],
                                         'elapsed_seconds': round(time.monotonic()-started, 3)})
        if response.status_code != 200:
            raise SemanticAPIError(f'Semantic API HTTP {response.status_code}; no retry')
        try:
            body = response.json()
            choice = body['choices'][0]
            raw_content = choice.get('message', {}).get('content')
            raw = {'prefix_sha256': request['prefix_sha256'], 'finish_reason': choice.get('finish_reason'),
                   'content': raw_content[:100000] if isinstance(raw_content, str) else None,
                   'storage_truncated': isinstance(raw_content, str) and len(raw_content) > 100000}
            self.raw_responses[request['prefix_sha256']] = raw
            if self.on_raw_response is not None:
                self.on_raw_response(dict(raw))
            usage = body.get('usage')
            self.metadata.append({'prefix_sha256': request['prefix_sha256'],
                                  'finish_reason': choice.get('finish_reason') if isinstance(choice.get('finish_reason'), str) else None,
                                  'usage': {k: v for k, v in usage.items()
                                            if k in ('prompt_tokens', 'completion_tokens', 'total_tokens')
                                            and type(v) is int} if isinstance(usage, dict) else {}})
            if choice.get('finish_reason') != 'stop':
                raise SemanticAPIError('Semantic API did not finish normally; packet rejected')
            content = choice['message']['content']
            if not isinstance(content, str):
                raise SemanticAPIError('Semantic API returned no text JSON packet')
            # Accept a single complete Markdown JSON wrapper, never search for
            # a JSON substring in prose/reasoning or repair a malformed packet.
            fenced = re.fullmatch(r'\s*```(?:json)?[ \t]*\r?\n(.*?)\r?\n```\s*',
                                  content, flags=re.DOTALL)
            packet = json.loads(fenced.group(1) if fenced else content)
            if not isinstance(packet, dict):
                raise SemanticAPIError('Semantic API packet must be a JSON object')
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            raise SemanticAPIError('Semantic API returned invalid JSON/envelope') from None
        self.response_packets[request['prefix_sha256']] = packet
        return packet

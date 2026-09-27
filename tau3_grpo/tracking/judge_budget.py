"""Shared budget ledger and cached DeepSeek requests for audits and evaluation.

Uses the pre-existing cumulative ledger, preserves confirmed actual billing,
and never retries an uncertain paid request automatically.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from tau3_grpo.models.semantic_api import OpenAICompatibleSemanticModel, SemanticAPIError
from tau3_grpo.utils.hashing import sha256_json

def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)



def flash_usage_estimate(usage, started_at, finished_at):
    """Verified 2026-09-26 rates; boundary uncertainty uses the higher rate.

This is still a usage estimate, never a provider invoice. Missing cache details
are treated as uncached input, not as free input.
"""
    def peak(stamp):
        local = datetime.fromisoformat(stamp).astimezone(ZoneInfo('Asia/Shanghai'))
        return local.weekday() < 5 and (9 <= local.hour < 12 or 14 <= local.hour < 18)
    factor = 1 if peak(started_at) or peak(finished_at) else 0.5
    inp, out = usage['prompt_tokens'], usage['completion_tokens']
    hit = usage.get('prompt_cache_hit_tokens')
    if hit is None:
        hit = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
    miss = usage.get('prompt_cache_miss_tokens')
    cache_valid = (type(hit) is int and 0 <= hit <= inp and
                   (miss is None or type(miss) is int and miss >= 0 and hit + miss == inp))
    hit = hit if cache_valid else 0
    cost = (hit * 0.04 + (inp - hit) * 2 + out * 8) * factor / 1_000_000
    return cost, {'time_band': 'peak' if factor == 1 else 'off_peak',
                  'cache_basis': 'reported_cache_usage' if cache_valid else 'uncached_upper_estimate',
                  'cache_hit_tokens_used': hit,
                  'rates_cny_per_million': {'cached_input': 0.04 * factor,
                                            'uncached_input': 2 * factor, 'output': 8 * factor}}


class Budget:
    """Reserve each call before sending; retain full reserve on ambiguous failure.

    Prices are conservative official CNY peak rates, treating all input as cache
    misses. Caller must hold the run lock; asyncio calls mutate this sequentially.
    """

    def __init__(self, path, limit=100.0, *, max_calls=200):
        self.path = Path(path)
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError('Invalid request cap')
        self.max_calls = max_calls
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state['limit_cny'] != limit:
                raise ValueError('Cannot change a run budget on resume')
        else:
            self.state = {'limit_cny': limit, 'currency': 'CNY', 'input_per_million': 2.0,
                          'output_per_million': 8.0, 'calls': [],
                          'price_source': 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing',
                          'price_verified_date': '2026-09-25', 'basis': 'peak_uncached_upper_estimate'}
            dump(self.path, self.state)

    @property
    def accounted(self):
        reconciliation = self.state.get('billing_reconciliation')
        if reconciliation is None:
            return sum(x['accounted_cny'] for x in self.state['calls'])
        n = reconciliation['through_call_count']
        if (len(self.state['calls']) < n or
                sha256_json(self.state['calls'][:n]) != reconciliation['prefix_sha256']):
            raise ValueError('Reconciled call history changed')
        return reconciliation['actual_cny'] + sum(x['accounted_cny'] for x in self.state['calls'][n:])

    def reconcile(self, actual_cny, *, source):
        """Caller holds the shared lock; preserve all historical estimates."""
        if type(actual_cny) not in (int, float) or not 0 <= actual_cny <= self.state['limit_cny']:
            raise ValueError('Invalid actual expense')
        old = self.state.get('billing_reconciliation')
        if old is not None:
            self.state.setdefault('billing_reconciliation_history', []).append(old)
        self.state['billing_reconciliation'] = {
            'actual_cny': actual_cny, 'source': source,
            'confirmed_at_utc': datetime.now(timezone.utc).isoformat(),
            'through_call_count': len(self.state['calls']),
            'prefix_sha256': sha256_json(self.state['calls']),
            'historical_estimate_cny': sum(x['accounted_cny'] for x in self.state['calls'])}
        self.state['settlement_policy'] = 'flash_time_cache_v2'
        self.state['price_verified_date'] = '2026-09-26'
        dump(self.path, self.state)

    def billing_summary(self):
        rec = self.state.get('billing_reconciliation')
        n = rec['through_call_count'] if rec else 0
        return {'confirmed_actual_cny': rec['actual_cny'] if rec else None,
                'confirmation_source': rec['source'] if rec else None,
                'new_usage_estimate_or_reservation_cny': sum(x['accounted_cny'] for x in self.state['calls'][n:]),
                'budget_committed_cny': self.accounted,
                'remaining_budget_cny': self.state['limit_cny'] - self.accounted,
                'is_provider_invoice': False}

    def reserve(self, request_id, system, payload, max_tokens):
        if any(r['request_id'] == request_id for r in self.state['calls']):
            raise ValueError('Duplicate budget request ID')
        if len(self.state['calls']) >= self.max_calls:
            raise ValueError('Pilot API request cap reached')
        # UTF-8 byte length plus conservative framing overhead, not chars/4.
        input_bound = len((system + json.dumps(payload, ensure_ascii=False)).encode()) + 4096
        bound = (input_bound * 2 + max_tokens * 8) / 1_000_000
        if self.accounted + bound > self.state['limit_cny']:
            raise ValueError('Budget cap: request was not sent')
        row = {'request_id': request_id, 'status': 'reserved', 'reserved_cny': bound,
               'accounted_cny': bound, 'input_bound': input_bound, 'output_bound': max_tokens,
               'started_at_utc': datetime.now(timezone.utc).isoformat()}
        self.state['calls'].append(row)
        dump(self.path, self.state)
        return row

    def settle(self, row, metadata, status):
        usage = metadata.get('usage', {})
        inp, out = usage.get('prompt_tokens'), usage.get('completion_tokens')
        row.update(status=status, usage=usage, response_model=metadata.get('response_model'))
        if type(inp) is int and type(out) is int and inp >= 0 and out >= 0:
            cost = (inp * 2 + out * 8) / 1_000_000
            if self.state.get('settlement_policy') == 'flash_time_cache_v2' and row.get('started_at_utc'):
                row['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
                cost, pricing = flash_usage_estimate(usage, row['started_at_utc'], row['finished_at_utc'])
                row['pricing'] = pricing
            row['accounted_cny'] = cost
            row['accounting'] = ('usage_at_time_cache_prices_not_invoice' if 'pricing' in row
                                 else 'usage_at_peak_uncached_prices_not_invoice')
            if cost > row['reserved_cny']:
                row['reservation_exceeded'] = True
        else:
            row['accounting'] = 'unknown_usage_full_reservation_retained'
        dump(self.path, self.state)
        if row.get('reservation_exceeded'):
            raise RuntimeError('Unexpected token bound exceeded; stop further calls')


async def call_json(output, budget, call_id, system, events, payload, max_tokens=6144, *, json_mode=False):
    path = output / 'calls' / f'{call_id}.json'
    request = {'system': system, 'visible_messages': events, 'prefix_sha256': sha256_json(events)}
    identity_fields = {'system': system, 'payload': payload, 'max_tokens': max_tokens,
                       'model': 'deepseek-flash', 'thinking': 'disabled'}
    if json_mode:
        identity_fields['response_format'] = {'type': 'json_object'}
    identity = sha256_json(identity_fields)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['request_hash'] != identity:
            raise ValueError('Cached request identity changed')
        if saved['status'] != 'received':
            raise ValueError('Previous attempt failed; no automatic paid retry')
        return saved['packet']
    if any(r['request_id'] == call_id for r in budget.state['calls']):
        raise ValueError('Unresolved prior reservation; no automatic paid retry')
    model = OpenAICompatibleSemanticModel.from_env(
        {'max_tokens': max_tokens, 'temperature': 0, 'timeout_seconds': 180,
         'thinking_mode': 'disabled', 'response_format_json': json_mode}, env_prefix='TAU3_JUDGE')
    if model.base_url not in ('https://api.deepseek.com', 'https://api.deepseek.com/v1') or model.model != 'deepseek-flash':
        raise ValueError('Pricing/model identity mismatch; no request sent')
    row = budget.reserve(call_id, system, payload, max_tokens)
    receipt = {'request_id': call_id, 'request_hash': identity, 'request': payload,
               'system_hash': sha256_json(system), 'client': model.provenance}
    # Preserve an in-flight request before any external call can be interrupted.
    dump(path, dict(receipt, status='reserved'))
    try:
        packet = await model.extract_json(request, user_payload=payload)
        receipt.update(status='received', packet=packet)
    except (SemanticAPIError, ValueError) as exc:
        receipt.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        metadata = model.metadata[-1] if model.metadata else {}
        receipt['response_metadata'] = metadata
        receipt['timings'] = model.request_timings
        receipt['raw_response'] = model.raw_responses.get(request['prefix_sha256'])
        dump(path, receipt)
        budget.settle(row, metadata, receipt.get('status', 'failed'))
    return packet


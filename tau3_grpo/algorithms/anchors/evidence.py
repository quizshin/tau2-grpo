"""Versioned decision evidence for live anchors; no inferred global consent.

V2 deliberately uses exact normalized dialogue evidence rather than claiming to
solve natural-language authorization. Different proposal/reply histories cannot
merge just because both contain 'yes'. This can split equivalent paraphrases;
measure cross-trajectory coverage before drawing conclusions about RL quality.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
import unicodedata
from typing import Any, Iterable

from tau3_grpo.algorithms.anchors.features import KNOWN_INFO_TOOLS
from tau3_grpo.utils.hashing import sha256_json

DEFAULT_ANCHOR_VERSION = 'v1'


def validate_version(value: str) -> str:
    if value not in ('v1', 'v2', 'v3'):
        raise ValueError(f'unsupported anchor version: {value!r}')
    return value


def _get(value: Any, key: str, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def text_value(value: Any) -> str:
    # Do not lowercase: identifiers and quoted values may be case-sensitive.
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else ''
    return ' '.join(unicodedata.normalize('NFC', value).split())


def json_value(value: Any):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value.strip()
    return value


def reply_kind(text: str) -> str:
    """Lexical telemetry only. Never used as an authorization decision."""
    text = text_value(text).lower().replace('’', "'")
    if not text:
        return 'none'
    if re.search(r"\b(?:not|no|don't|cannot|can't|wait|withdraw|revoke|hold on)\b", text):
        return 'negation_or_pause_text'
    if '?' in text or re.search(r'\b(?:if|once|after|before|until)\b', text):
        return 'conditional_or_question_text'
    if re.fullmatch(r'(?:yes(?:,? please)?(?:,? (?:go ahead|proceed))?|i confirm|go ahead|proceed)[.!]?', text):
        return 'assent_text'
    return 'other_text'


@dataclass(frozen=True)
class DecisionEvidence:
    read_hash: str
    dialogue_hash: str
    tool_event_hash: str
    proposal_hash: str | None
    reply_hash: str | None
    reply_kind: str
    committed_tools: tuple[str, ...]
    read_keys: int
    version: str = "v2"
    normalized_utterances: int = 0
    opaque_utterances: int = 0

    def payload(self):
        return {'version': self.version, 'read_hash': self.read_hash, 'dialogue_hash': self.dialogue_hash,
                'tool_event_hash': self.tool_event_hash,
                'confirmation_exchange': {'proposal_hash': self.proposal_hash, 'reply_hash': self.reply_hash},
                'committed_tools': self.committed_tools}

    @property
    def digest(self):
        return sha256_json(self.payload())


def decision_evidence(messages: Iterable[Any], *, version: str = "v2") -> DecisionEvidence:
    """Reconstruct only observed evidence. No future action/reward/DB peeking.

    Successful reads form a sorted query ledger. Repeating the same observation
    is idempotent; changed observations for a query retain their order. Writes,
    failures and unmatched results are ordered events. Tool-call IDs are used
    solely to join requests to responses, never as state features.
    """
    validate_version(version)
    if version == "v1":
        raise ValueError("v1 uses the legacy feature extractor")
    normalized_count = opaque_count = 0
    pending = {}
    reads: dict[str, list[str]] = {}
    dialogue = sha256_json([])
    events = sha256_json([])
    proposal = reply = None
    kind = 'none'
    committed = set()
    for message in messages:
        role = str(_get(message, 'role', ''))
        text = text_value(_get(message, 'content'))
        if role in ('assistant', 'user') and text:
            # Full visible conversational evidence is the conservative fallback
            # for unresolved scope/conditions, not a guessed action parser.
            normalized = text
            if version == "v3":
                from tau3_grpo.algorithms.anchors.semantic import normalize_utterance
                normalized = normalize_utterance(role, text)
                normalized_count += int(normalized["kind"] != "opaque")
                opaque_count += int(normalized["kind"] == "opaque")
            dialogue = sha256_json([dialogue, role, normalized])
            if role == 'assistant':
                proposal = sha256_json({'text': normalized, 'reads': reads, 'events': events})
                reply = None
                kind = 'none'
            else:
                reply = sha256_json(normalized)
                kind = reply_kind(text)
        for call in _get(message, 'tool_calls', None) or []:
            function = _get(call, 'function', None)
            name = _get(call, 'name') or _get(function, 'name')
            arguments = _get(call, 'arguments', None)
            if arguments is None:
                arguments = _get(function, 'arguments')
            call_id = str(_get(call, 'id', ''))
            if call_id in pending:
                raise ValueError('duplicate outstanding tool call id in anchor evidence')
            pending[call_id] = {'name': name, 'arguments': json_value(arguments)}
        if role == 'tool':
            call_id = str(_get(message, 'id', None) or _get(message, 'tool_call_id', ''))
            call = pending.pop(call_id, None)
            result = json_value(_get(message, 'content'))
            failed = bool(_get(message, 'error', False))
            event = {'call': call, 'result': result, 'error': failed}
            if call and call['name'] in KNOWN_INFO_TOOLS and not failed:
                key = sha256_json(call)
                result_hash = sha256_json(result)
                history = reads.setdefault(key, [])
                if not history or history[-1] != result_hash:
                    history.append(result_hash)
            else:
                events = sha256_json([events, event])
                if call and not failed:
                    committed.add(str(call['name']))
            # Tool evidence changes invalidate the equality of the overall
            # decision state, without making a semantic claim about consent.
    if pending:
        # Normal sequential rollouts complete a tool batch before generation.
        # Preserve incomplete evidence instead of collapsing it into no action.
        events = sha256_json([events, 'pending', sorted(pending.values(), key=sha256_json)])
    return DecisionEvidence(sha256_json(reads), dialogue, events, proposal, reply, kind,
                            tuple(sorted(committed)), len(reads), version, normalized_count, opaque_count)

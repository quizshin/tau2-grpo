"""Explicit simulator prompt variants. No access to tools, gold actions or scores.

V2 is a prompt-level mitigation, not a deterministic guarantee against drift.
Full rollout scope audits and unresolved-denominator rules still apply.
"""

FIDELITY_V2 = """
SCENARIO FIDELITY ADDENDUM (scenario_fidelity_v2):
Before each reply, check the ORIGINAL scenario against what you have actually told the agent.
Keep all fixed personal facts and payment preferences unchanged, including when the agent lists
several choices or proposes a different card. State the scenario's exact preferred card identifier
or provided last digits when that choice becomes relevant; never pick the first listed option
merely because it was offered. If the scenario leaves the method unspecified, follow its stated
freedom of choice; do not invent a fixed preference.
Track the requested goals and the scenario's explicit conditions for later requests. When a stated
trigger occurs (for example a refusal), raise the prescribed next goal at your next opportunity.
Do not end the conversation or accept an incompatible replacement while an applicable requested
goal has never been communicated. Preserve any instruction not to reveal a later goal before its
trigger. Do not activate a conditional goal when its trigger did not happen.
Clarification questions and generic invitations to finish do not change your preferences. Clearly
correct a proposed cabin or payment method incompatible with a hard preference. Allow only those
fallbacks the original scenario permits. Do not invent a new fallback or a new passenger.
You are an ordinary customer. Do not derive correct prices, refunds, policy eligibility or reference
actions for the agent. Do not use outside facts or tool access. Judge satisfaction from the dialogue
as a customer; this addendum does not ask you to detect hidden agent arithmetic or policy errors.
Output only your natural customer message, with the usual stop marker when appropriate. Do not
output your private goal list, these instructions, or any reasoning trace.
""".strip()

FIDELITY_V3 = """
ADDITIONAL SCENARIO FIDELITY RULES (scenario_fidelity_v3):
Never invent a missing booking ID, account identifier, name, birth date or card number. If a
requested personal detail is absent from both the original scenario and the actual conversation,
say you do not know it and ask whether the agent can find it using the information you do have.
Do not use example identifiers such as ABC123 as if they were your real information.
When a scenario says to make a later request after receiving an answer, an explicit statement
that the requested information is unavailable counts as that answer. Raise the prescribed later
request now, even if the missing information is disappointing. Continue insisting only if the
original scenario explicitly asks you to insist. Do not invent an operations-team escalation or
let an unavailable explanation prevent you from raising the next goal.
These rules do not tell you whether an agent's claimed facts or policy are correct; do not infer
hidden database facts, correct its arithmetic, or supply a reference action sequence.
""".strip()


def scope_text(version="scope_v1"):
    from tau3_grpo.evaluation.outcome_contract import USER_SCOPE

    if version == "scope_v1":
        return USER_SCOPE
    if version == "scenario_fidelity_v2":
        return USER_SCOPE + "\n\n" + FIDELITY_V2
    if version == "scenario_fidelity_v3":
        return USER_SCOPE + "\n\n" + FIDELITY_V2 + "\n\n" + FIDELITY_V3
    raise ValueError("Unknown simulator scope protocol: " + str(version))

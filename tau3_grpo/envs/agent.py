"""Official LLMAgent behavior with the project's explicit multi-call prompt."""

from tau2.agent.llm_agent import LLMAgent

from tau3_grpo.prompts import build_system_prompt


class MultiCallAirlineAgent(LLMAgent):
    @property
    def system_prompt(self) -> str:
        return build_system_prompt(self.domain_policy)

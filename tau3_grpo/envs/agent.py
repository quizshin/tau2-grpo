"""Official LLMAgent behavior with the project's explicit multi-call prompt."""

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import MultiToolMessage, ToolMessage

from tau3_grpo.envs.observations import project_tool_text
from tau3_grpo.prompts import build_system_prompt


class MultiCallAirlineAgent(LLMAgent):
    def __init__(self, *args, project_observations: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_observations = project_observations
        self.observation_receipts = []

    def _generate_next_message(self, message, state):
        if not self.project_observations:
            return super()._generate_next_message(message, state)
        # The official orchestrator and this persistent state keep raw messages.
        # Only the ephemeral model request receives clipped copies.
        incoming = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        view = state.model_copy(deep=True)
        for item in view.messages:
            if isinstance(item, ToolMessage):
                item.content, _ = project_tool_text(item.content, 65536, "middle")
        projected = message.model_copy(deep=True)
        visible = projected.tool_messages if isinstance(projected, MultiToolMessage) else [projected]
        for item in visible:
            if isinstance(item, ToolMessage):
                item.content, receipt = project_tool_text(item.content, 65536, "middle")
                self.observation_receipts.append({"tool_call_id": item.id, **receipt})
        answer = super()._generate_next_message(projected, view)
        state.messages.extend(incoming)
        return answer

    @property
    def system_prompt(self) -> str:
        return build_system_prompt(self.domain_policy)

import logging
from collections.abc import Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from sb3_api.agent.callbacks.recursion_tracker import RecursionTracker
from sb3_api.agent.prompts.prompts import PromptId, PromptRegistry
from sb3_api.models.constants import FIRST_RECURSION_THRESHOLD, RECURSION_LIMIT

logger = logging.getLogger(__name__)


class RecursionLimitHandler(AgentMiddleware):
    """Middleware to handle recursion limit by restricting tools at threshold."""

    def __init__(
        self,
        callback: RecursionTracker,
        prompt_registry: PromptRegistry,
    ) -> None:
        self.callback = callback
        self.prompt_registry = prompt_registry
        self.warning_sent = False

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        current_step = self.callback.current_step

        if current_step >= FIRST_RECURSION_THRESHOLD and not self.warning_sent:
            logger.warning("Recursion threshold reached: %d/%d", current_step, RECURSION_LIMIT)
            remaining_steps = self.callback.get_remaining_steps()
            wrn_template = self.prompt_registry.get_prompt(PromptId.RECURSION_WARNING)
            wrn_message = wrn_template.format(
                current_step=current_step,
                recursion_limit=RECURSION_LIMIT,
                remaining_steps=remaining_steps,
            )

            wrn_prompt = (request.system_prompt or "") + wrn_message
            request = request.override(system_prompt=wrn_prompt)
            self.warning_sent = True

        if current_step >= RECURSION_LIMIT - 4:
            logger.warning("Forcing partial_results tool at step %d", current_step)
            request = request.override(tool_choice="partial_results_generator")
        return handler(request)

import logging

from sb3_api.models.constants import RECURSION_LIMIT

logger = logging.getLogger(__name__)


class RecursionTracker:
    """Tracks agent step count and partial_results tool output.

    No longer extends LangChain BaseCallbackHandler.
    Step count is incremented externally by the agent executor.
    """

    def __init__(self) -> None:
        self.current_step: int = 0
        self.partial_results: str | None = None

    def increment_step(self) -> None:
        self.current_step += 1

    def record_partial_results(self, content: str) -> None:
        self.partial_results = content

    def get_remaining_steps(self) -> int:
        return RECURSION_LIMIT - self.current_step

    def reset(self) -> None:
        self.current_step = 0
        self.partial_results = None
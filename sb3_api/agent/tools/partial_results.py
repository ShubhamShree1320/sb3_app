import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PartialResultsInput(BaseModel):
    partial_content: str = Field(description="All work/reasoning accomplished so far")
    query: str = Field(description="The original user query")


class PartialResultsTool(BaseTool):
    """Tool for generating partial results with recommendations when approaching recursion limit."""

    name: str = "partial_results_generator"
    description: str = (
        "Use this when approaching processing limits to provide partial results. "
        "Summarize what you have accomplished so far and ensure the user gets valuable partial results"
    )
    args_schema: type[BaseModel] = PartialResultsInput
    llm: Any = Field(description="LLM chain for generating comprehensive recommendations")

    def _run(self, partial_content: str, query: str) -> dict:
        prompt_input = {
            "partial_content": partial_content,
            "current_query": query,
        }
        response = self.llm.invoke(prompt_input)
        return response.content

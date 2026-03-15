import json
import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from sb3_api.enums.trace import TraceType


class _GeneratePlotInput(BaseModel):
    query: str = Field(description="The user query")
    sql_results: str = Field(description="The SQL query results")
    plot_type: str = Field(description="The determined plot type")


class GeneratePlotTool(BaseTool):
    """Tool for generating plot JSON based on query, data, and plot type."""

    name: str = "generate_plot"
    description: str = (
        "Use this tool to generate visualization JSON format from query, data, plot type and plot type reason"
    )
    args_schema: type[BaseModel] = _GeneratePlotInput
    llm: Runnable

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tags: list[str] = [TraceType.PLOT]

    def _run(
        self,
        query: str,
        sql_results: str,
        plot_type: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict:
        raw_response = self.llm.invoke(
            {
                "query": query,
                "results": sql_results,
                "plot_type": plot_type,
            }
        )
        response = self._clean_json_response(raw_response.content.strip())
        try:
            plot_json = json.loads(response)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse generated JSON: {e}"}
        else:
            return plot_json

    @staticmethod
    def _clean_json_response(response: str) -> str:
        """Clean JSON response by removing markdown formatting and trailing commas."""
        if "```" in response:
            match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
            if match:
                response = match.group(1)
            else:
                response = re.sub(r"^```(?:json)?\s*", "", response)
                response = re.sub(r"```\s*$", "", response)

        brace_index = response.find("{")
        if brace_index != -1:
            response = response[brace_index:]
        response = re.sub(r",(\s*[}\]])", r"\1", response)
        return response.strip()

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class _PlotTypeInput(BaseModel):
    query: str = Field(description="The user query")
    sql_results: str = Field(description="The SQL query results")


class DeterminePlotTypeTool(BaseTool):
    """Tool for determining the appropriate plot type based on query and data."""

    name: str = "determine_plot_type"
    description: str = (
        "Use this tool to generate visualizations when data is present or requested. Analyze the"
        "conversation history and current context to identify the relevant data if possible, "
        "then determine and create the most appropriate visualization type as a parseable JSON file. "
        "This tool should be called whenever the user explicitly requests a plot (e.g., 'make a plot', "
        "'visualize this', 'create a chart') or when visualization would enhance understanding of "
        "the data being discussed."
    )
    args_schema: type[BaseModel] = _PlotTypeInput
    llm: Runnable

    def _run(
        self,
        query: str,
        sql_results: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict:
        # Guard against all empty/no-data result formats.
        # LangChain SQLDatabase returns "" for 0 rows (not "[]"), so the original
        # exact-match check missed it and the LLM hallucinated a chart type.
        _s = (sql_results or "").strip()
        if not _s or _s in ("[]", "None", "null") or "(0 rows)" in _s.lower():
            return {"plot_type": None, "reason": "No data available"}

        response = self.llm.invoke({"query": query, "results": sql_results})

        lines = response.content.split("\n")
        plot_type = lines[0].split(":")[1].strip()
        reason = lines[1] if len(lines) > 1 else ""

        return {"plot_type": plot_type, "reason": reason}

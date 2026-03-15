import logging
from typing import Any

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from sb3_api.enums.trace import TraceType

logger = logging.getLogger(__name__)


class AdaptQueryInput(BaseModel):
    user_question: str = Field(description="The new user question to answer")
    query_examples: str = Field(
        description="Example SQL queries that were used to answer similar questions in the past"
    )
    business_context: str = Field(
        default="",
        description=("Business context including KPI definitions and table schemas."),
    )

    @field_validator("query_examples", mode="before")
    @classmethod
    def validate_query_examples(cls, v: str) -> str:
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                raise ValueError("query_examples cannot be empty or whitespace")
            return stripped
        return v


class AdaptQueryTool(BaseTool):
    """Tool for adapting past SQL queries to answer new similar questions."""

    name: str = "adapt_query_from_examples"
    description: str = (
        "This tool adapts the SQL query from similar past queries to the provided user question. "
        "Use this tool when the context contains example past queries."
    )
    args_schema: type[BaseModel] = AdaptQueryInput
    llm: Runnable

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tags: list[str] = [TraceType.SQL]

    def _run(
        self,
        user_question: str,
        query_examples: str,
        business_context: str = "",
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        logger.info("Adapting query from similar past examples: %s...", query_examples[:200])

        try:
            response = self.llm.invoke(
                {
                    "user_question": user_question,
                    "query_examples": query_examples,
                    "business_context": business_context,
                }
            )
        except Exception:
            logger.exception("Failed to adapt query")
            return "Query adapting failed. Proceed with standard query building approach."

        result = response.content.strip()

        if not result:
            logger.info("Query examples not applicable for adapting: %s", user_question)
            return "Retrieved examples are not applicable to the current question."

        logger.info("Successfully adapted query from examples")
        return f"[Adapted query from similar past queries]\n{result}"

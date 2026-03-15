from collections.abc import Sequence
from typing import Any, Literal

from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
)
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.engine import Result

from sb3_api.enums.trace import TraceType
from sb3_api.repository.agent.base import BaseSQLDatabase


class BaseRedshiftSQLDatabaseTool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    db: BaseSQLDatabase = Field(exclude=True)


class ListRedshiftSQLDatabaseTool(BaseRedshiftSQLDatabaseTool, ListSQLDatabaseTool):  # type: ignore[misc]
    pass


class InfoRedshiftSQLDatabaseTool(BaseRedshiftSQLDatabaseTool, InfoSQLDatabaseTool):  # type: ignore[misc]
    pass


class QueryRedshiftSQLCheckerTool(BaseRedshiftSQLDatabaseTool, QuerySQLCheckerTool):  # type: ignore[misc]
    pass


class _QuerySQLDatabaseToolInput(BaseModel):
    query: str = Field(..., description="A detailed and correct SQL query.")
    query_purpose: Literal["primary", "clarification"] = Field(
        ...,
        description=(
            "Categorize this query's purpose:\n"
            "- 'primary': This query directly answers the initial user's question even if the data are not available.\n"
            "- 'clarification': This query explores the database or checks data availability\n\n"
            "Mark as 'clarification' if: checking DISTINCT values, using LIMIT 1-10 for exploration, "
            "verifying data exists, understanding schema or derivations of primary query"
            "but adapted for the available data\n"
            "Mark as 'primary' if: this is the final query that would answer the initial user query"
        ),
    )


class QueryRedshiftSQLDatabaseTool(BaseRedshiftSQLDatabaseTool, BaseTool):
    """SQL query tool that tracks queries via metadata in callbacks."""

    name: str = "sql_db_query"
    description: str = """
    Execute a SQL query against the database and retrieve the result.
    """
    args_schema: type[BaseModel] = _QuerySQLDatabaseToolInput

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.tags: list[str] = [TraceType.SQL]  # override the basetool variable tags that is forwarded to callbacks

    def _run(
        self,
        query: str,
        query_purpose: Literal["primary", "clarification"],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str | Sequence[dict[str, Any]] | Result:
        result = self.db.run_no_throw(query)

        if run_manager and query_purpose == "primary":
            run_manager.metadata["primary"] = True

        return result

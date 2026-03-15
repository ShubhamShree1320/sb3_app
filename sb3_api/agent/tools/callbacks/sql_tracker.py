# from typing import Any
#
# from langchain_core.callbacks import BaseCallbackHandler
#
#
# class SQLQueryTracker(BaseCallbackHandler):
#     """Features:
#     - Captures the primary query that answers the initial user's query
#     - Captures the result of the primary query.
#     """
#
#     def __init__(self) -> None:
#         super().__init__()
#
#         self._primary_run_id: Any | None = None
#         self._primary_query: str | None = None
#         self._primary_query_result: Any | None = None
#
#     def on_tool_start(  # noqa: PLR0913
#         self,
#         serialized: dict[str, Any],
#         input_str: str,
#         *,
#         run_id: Any,
#         parent_run_id: Any | None = None,
#         tags: list[str] | None = None,
#         metadata: dict[str, Any] | None = None,
#         **kwargs: Any,
#     ) -> None:
#         tool_name = serialized.get("name")
#
#         if tool_name != "sql_db_query":
#             return
#
#         query_purpose = kwargs.get("inputs", {}).get("query_purpose")
#
#         if query_purpose == "primary" and self._primary_query is None:
#             self._primary_query = kwargs["inputs"].get("query")
#             self._primary_run_id = run_id
#
#     def on_tool_end(
#         self,
#         output: Any,
#         *,
#         run_id: Any,
#         parent_run_id: Any | None = None,
#         tags: list[str] | None = None,
#         **kwargs: Any,
#     ) -> None:
#         if self._primary_run_id is not None and run_id == self._primary_run_id:
#             self._primary_query_result = output  # for future if needed
#
#     def get_primary_query(self) -> str | None:
#         return self._primary_query
#
#     def get_query_result(self) -> Any | None:
#         return self._primary_query_result
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SQLQueryTracker:
    """Captures the primary SQL query that answers the user's question.

    No longer extends LangChain BaseCallbackHandler.
    The primary query is captured directly when the SQL query tool is called
    with query_purpose='primary'.
    """

    def __init__(self) -> None:
        self._primary_query: str | None = None
        self._primary_query_result: Any | None = None

    def record_query(self, query: str, query_purpose: str, result: Any = None) -> None:
        """Record a SQL query. Always keeps the LAST primary query.

        Deliberately overwrites any earlier primary query. When the LLM makes
        a mistake on the first attempt (wrong column, wrong syntax) and
        self-corrects, we want to show the corrected query — not the failed one.
        SQLQueryTracker is reset at the start of every request in _invoke_and_parse
        so there is no cross-request contamination.
        """
        if query_purpose == "primary":
            self._primary_query = query
            self._primary_query_result = result
            logger.debug("Captured primary SQL query (last wins): %s", query[:100])

    def get_primary_query(self) -> str | None:
        return self._primary_query

    def get_query_result(self) -> Any | None:
        return self._primary_query_result
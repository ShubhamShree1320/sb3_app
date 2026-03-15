from collections.abc import Iterable, Sequence
from typing import Any, Literal
from langchain_community.utilities import SQLDatabase
from sqlalchemy import Result
from sb3_api.repository.agent.base import BaseSQLDatabase

class SQLDatabaseWrapper(BaseSQLDatabase):
    def __init__(self, sql_db: SQLDatabase) -> None:
        self._sql_db = sql_db

    def get_connection(self):
        return self._sql_db._engine.raw_connection()

    def get_schema(self) -> str:
        return "public"

    def get_table_info(self, table_names: list[str] | None = None) -> str:
        return self._sql_db.get_table_info(table_names)

    def run(self, command: str, fetch: Literal["all", "one"] = "all") -> Any:
        return self._sql_db.run(command, fetch)

    def get_usable_table_names(self) -> Iterable[str]:
        return self._sql_db.get_usable_table_names()

    def run_no_throw(
        self, command: str, fetch: Literal["all", "one"] = "all"
    ) -> str | Sequence[dict[str, Any]] | Result[Any]:
        return self._sql_db.run_no_throw(command, fetch)

    def get_table_info_no_throw(self, table_names: list[str] | None = None) -> str:
        return self._sql_db.get_table_info_no_throw(table_names)

    @property
    def dialect(self) -> str:
        return self._sql_db.dialect
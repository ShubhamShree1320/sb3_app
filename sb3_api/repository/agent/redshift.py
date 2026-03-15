# ruff: noqa: BLE001

import contextlib
import logging
from collections.abc import Generator, Iterable, Sequence
from typing import Any, Literal

import redshift_connector
from redshift_connector import Connection
from sqlalchemy import Result, select, table, text
from sqlalchemy.sql.expression import TableClause

from sb3_api.repository.agent.base import BaseSQLDatabase
from sb3_api.repository.agent.redshift_connection import RedshiftConnectionPool
from sb3_api.utils.utils import truncate_string, validate_sql_identifier

logger = logging.getLogger(__name__)


class RedshiftSQLDatabase(BaseSQLDatabase):
    def __init__(  # noqa: PLR0913
        self,
        connection_pool: RedshiftConnectionPool,
        schema: str | None = None,
        ignore_tables: list[str] | None = None,
        include_tables: list[str] | None = None,
        sample_rows_in_table_info: int = 3,
        max_string_length: int = 10_000,
    ) -> None:
        """SQLDatabase from Langchain with connection pooling."""
        self._pool = connection_pool
        self._schema = schema

        self._all_tables = self._get_all_tables()

        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                msg = f"include_tables {missing_tables} not found in database"
                raise ValueError(msg)
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                msg = f"ignore_tables {missing_tables} not found in database"
                raise ValueError(msg)
        usable_tables = self.get_usable_table_names()
        self._usable_tables = set(usable_tables) if usable_tables else self._all_tables
        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._max_string_length = max_string_length

    def _get_all_tables(self) -> set[str]:
        cursor = self._pool.connection.cursor()
        try:
            cursor.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                AND table_type = 'BASE TABLE'
            """,
                (self._schema or "public",),
            )

            return {row[0] for row in cursor.fetchall()}
        finally:
            cursor.close()

    def get_schema(self) -> str | None:
        return self._schema

    def get_connection(self) -> Connection:
        return self._pool.connection

    def get_usable_table_names(self) -> Iterable[str]:
        if self._include_tables:
            return sorted(self._include_tables)
        return sorted(self._all_tables - self._ignore_tables)

    def get_table_info(self, table_names: list[str] | None = None) -> str:
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                msg = f"table_names {missing_tables} not found in database"
                raise ValueError(msg)
            all_table_names = table_names

        conn = self._pool.connection
        tables = []
        for table_name in all_table_names:
            table_info = self._get_table_info(table_name, conn)
            tables.append(table_info)

        tables.sort()
        return "\n\n".join(tables)

    def _get_table_info(self, table_name: str, conn: Connection | None = None) -> str:
        if conn is None:
            conn = self._pool.connection
        cursor = conn.cursor()
        try:
            validated_table = validate_sql_identifier(table_name)
            schema = self._schema or "public"
            validated_schema = validate_sql_identifier(schema)

            # Get column information
            cursor.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = %s
            """,
                (validated_table, validated_schema),
            )
            columns = cursor.fetchall()

            # Create table info
            quoted_table = f'"{validated_schema}"."{validated_table}"'
            table_info = f"CREATE TABLE {quoted_table} (\n"
            table_info += ",\n".join(f"    {col[0]} {col[1]}" for col in columns)
            table_info += "\n);"

            # Add sample rows if specified
            if self._sample_rows_in_table_info > 0:
                table_obj = table(validated_table, schema=validated_schema)
                sample_rows = self._get_sample_rows(table_obj, conn)
                if sample_rows:
                    table_info += f"\n\n/* {self._sample_rows_in_table_info} rows from {quoted_table} table:\n"
                    table_info += "\n".join(str(row) for row in sample_rows)
                    table_info += "\n*/"

            return table_info
        finally:
            cursor.close()

    def _get_sample_rows(self, table_obj: TableClause, conn: Connection | None = None) -> list[dict[str, Any]]:
        if conn is None:
            conn = self._pool.connection
        cursor = conn.cursor()
        try:
            query = select(text("*")).select_from(table_obj).limit(self._sample_rows_in_table_info)
            cursor.execute(str(query.compile(compile_kwargs={"literal_binds": True})))
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row, strict=False)) for row in rows]
        finally:
            cursor.close()

    def run(self, command: str, fetch: Literal["all", "one"] = "all") -> list[str]:
        cursor = self._pool.connection.cursor()
        try:
            cursor.execute(command)
        except (redshift_connector.InterfaceError, redshift_connector.OperationalError) as e:
            cursor.close()  # Close the failed cursor before retry
            idx = self._pool.reset_last_connection()
            logger.warning(
                "Redshift connection error - resetting and retrying",
                extra={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "connection_index": idx,
                    "query_preview": command[:200],
                    "event": "connection_reset",
                },
            )
            cursor = self._pool.connection.cursor()
            cursor.execute(command)

        try:
            results = cursor.fetchall() if fetch == "all" else [cursor.fetchone()]

            columns = [desc[0] for desc in cursor.description]
            return [
                str(
                    {
                        col: truncate_string(value, self._max_string_length)
                        for col, value in zip(columns, row, strict=False)
                    }
                )
                for row in results
            ]
        finally:
            cursor.close()

    @contextlib.contextmanager
    def run_with_cursor(self, command: str) -> Generator[redshift_connector.Cursor]:
        """Execute a query and yield the cursor, ensuring cleanup."""
        cursor = self._pool.connection.cursor()
        try:
            cursor.execute(command)
            yield cursor
        finally:
            cursor.close()

    def run_no_throw(
        self, command: str, fetch: Literal["all", "one"] = "all"
    ) -> str | Sequence[dict[str, Any]] | Result[Any]:
        try:
            result = self.run(command, fetch)
            if not result:
                return ""
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    def get_table_info_no_throw(self, table_names: list[str] | None = None) -> str:
        try:
            return self.get_table_info(table_names)
        except ValueError as e:
            return f"Error: {e}"

    def get_context(self) -> dict[str, Any]:
        table_names = list(self.get_usable_table_names())
        table_info = self.get_table_info_no_throw()
        return {"table_info": table_info, "table_names": ", ".join(table_names)}

    @property
    def dialect(self) -> str:
        return "postgresql"

import datetime
import decimal
import logging
import time
from typing import TYPE_CHECKING, Any

from sqlalchemy import case, column, func, literal, select, table
from sqlalchemy.sql.expression import TableClause

if TYPE_CHECKING:
    from sqlalchemy.sql.selectable import Select

from sb3_api.repository.agent.redshift import RedshiftSQLDatabase
from sb3_api.utils.utils import validate_sql_identifier

logger = logging.getLogger(__name__)


class Profiler:
    def __init__(self, db: RedshiftSQLDatabase, schema: str | None = None) -> None:
        self._db = db
        self._schema = schema or db.get_schema() or "public"

    def _get_table_obj(self, table_name: str) -> TableClause:
        validated_schema = validate_sql_identifier(self._schema)
        validated_table = validate_sql_identifier(table_name)
        return table(validated_table, schema=validated_schema)

    def _compile_query(self, query: Any) -> str:
        return str(query.compile(compile_kwargs={"literal_binds": True}))

    def _get_columns(self, table_name: str) -> list[tuple[str, str]]:
        existing_tables = self._db.get_usable_table_names()

        if table_name not in existing_tables:
            msg = f"Table '{table_name}' does not exist in the database."
            raise ValueError(msg)

        cursor = self._db.get_connection().cursor()
        try:
            cursor.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = %s
                ORDER BY ordinal_position
                """,
                (table_name, self._schema),
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def compute_table_statistics(self, table_name: str) -> dict[str, Any]:
        logger.info("Computing statistics for table: %s", table_name)
        start = time.time()
        cursor = self._db.get_connection().cursor()
        try:
            columns = self._get_columns(table_name)
            col_names = [c[0] for c in columns]
            table_obj = self._get_table_obj(table_name)

            # Count rows
            query = select(func.count()).select_from(table_obj)
            cursor.execute(self._compile_query(query))
            number_of_rows = cursor.fetchone()[0]

            # Missing values
            col_exprs = []
            for col_name in col_names:
                case_expr = case((column(col_name).is_(None), literal(1)), else_=literal(0))
                sum_expr = func.sum(case_expr).label(f"{col_name}_missing")
                col_exprs.append(sum_expr)

            query = select(*col_exprs).select_from(table_obj)
            cursor.execute(self._compile_query(query))
            row = cursor.fetchone()
            missing_values = {col: row[idx] for idx, col in enumerate(col_names)}

            columns_statistics: dict[str, dict[str, Any]] = {}

            for col, data_type in columns:
                stats: dict[str, Any] = {}
                if any(t in data_type.lower() for t in ("int", "decimal", "numeric", "double", "real", "float")):
                    stats.update(self.process_numeric_statistics(col, table_name))
                elif any(t in data_type.lower() for t in ("date", "timestamp", "time")):
                    stats.update(self.process_datetime_statistics(col, table_name))
                else:
                    stats.update(self.process_categorical_statistics(col, table_name))

                stats.update(
                    {
                        "missing_values": missing_values[col],
                        "data_type": data_type,
                    }
                )

                columns_statistics[col] = stats

            general_statistics = {
                "number_of_rows": number_of_rows,
                "missing_values": missing_values,
            }

            result = {
                "general_statistics": general_statistics,
                "columns_statistics": columns_statistics,
            }
            logger.info("Computed statistics for table %s in %.2fs", table_name, time.time() - start)
            return self.make_json_serializable(result)
        finally:
            cursor.close()

    def process_numeric_statistics(self, col: str, table_name: str) -> dict[str, Any]:
        cursor = self._db.get_connection().cursor()
        try:
            table_obj = self._get_table_obj(table_name)
            query = select(
                func.min(column(col)),
                func.max(column(col)),
                func.avg(column(col)),
                func.percentile_cont(0.5).within_group(column(col)),
                func.stddev_pop(column(col)),
            ).select_from(table_obj)
            cursor.execute(self._compile_query(query))
            row = cursor.fetchone()
            return {
                "min_value": row[0],
                "max_value": row[1],
                "mean_value": row[2],
                "median_value": row[3],
                "std_value": row[4],
            }
        finally:
            cursor.close()

    def process_datetime_statistics(self, col: str, table_name: str) -> dict[str, Any]:
        cursor = self._db.get_connection().cursor()
        try:
            table_obj = self._get_table_obj(table_name)
            query: Select[Any] = select(func.min(column(col)), func.max(column(col))).select_from(table_obj)
            cursor.execute(self._compile_query(query))
            row = cursor.fetchone()
            return {
                "min_value": row[0],
                "max_value": row[1],
            }
        finally:
            cursor.close()

    def process_categorical_statistics(self, col: str, table_name: str) -> dict[str, Any]:
        cursor = self._db.get_connection().cursor()
        try:
            table_obj = self._get_table_obj(table_name)
            query: Select[Any] = select(func.count(func.distinct(column(col)))).select_from(table_obj)
            cursor.execute(self._compile_query(query))
            nb_unique_values = cursor.fetchone()[0]

            query = select(column(col)).distinct().limit(30).select_from(table_obj)
            cursor.execute(self._compile_query(query))
            unique_values = [r[0] for r in cursor.fetchall()]

            return {
                "unique_values": unique_values,
                "nb_unique_values": nb_unique_values,
            }
        finally:
            cursor.close()

    def make_json_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.make_json_serializable(v) for v in obj]
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return obj

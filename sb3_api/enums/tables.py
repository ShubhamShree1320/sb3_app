from enum import Enum


class TableName(str, Enum):
    FEEDBACK = "feedback"
    QUERY_RECORDS = "query-records"

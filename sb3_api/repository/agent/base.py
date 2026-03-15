from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from sqlalchemy import Result


class BaseSQLDatabase(ABC):
    @abstractmethod
    def get_usable_table_names(self) -> Iterable[str]:
        pass

    @abstractmethod
    def get_table_info(self, table_names: list[str] | None = None) -> str:
        pass

    @abstractmethod
    def run(self, command: str, fetch: Literal["all", "one"] = "all") -> Any:
        pass

    @abstractmethod
    def run_no_throw(
        self, command: str, fetch: Literal["all", "one"] = "all"
    ) -> str | Sequence[dict[str, Any]] | Result[Any]:
        pass

    @abstractmethod
    def get_table_info_no_throw(self, table_names: list[str] | None = None) -> str:
        pass

    @property
    @abstractmethod
    def dialect(self) -> str:
        pass

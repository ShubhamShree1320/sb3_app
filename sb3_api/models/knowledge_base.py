from typing import Any, Literal

from langchain_core.documents import Document
from pydantic import BaseModel

from sb3_api.enums.document import DocumentType

SIMILARITY_FORMAT_TEMPLATE = "[Similarity: {score}] \n{content}"
DOCUMENT_SEPARATOR = "\n\n---\n\n"


class SearchResult(BaseModel):
    document: Document
    score: float


class DocumentContext(BaseModel):
    acronyms: str | None = None
    context: list[SearchResult]
    doc_type: DocumentType | None = None

    def _format_table_statistics(self, stats: dict[str, Any]) -> str:
        general = stats.get("general_statistics", {}) or {}
        columns = stats.get("columns_statistics", {}) or {}

        lines = []
        lines.append("Table statistics:")

        if general:
            lines.append("  General:")
            number_of_rows = general.get("number_of_rows")
            if number_of_rows is not None:
                lines.append(f"    - number_of_rows: {number_of_rows}")

            missing_values = general.get("missing_values") or {}
            if missing_values:
                lines.append("    - missing_values_by_column:")
                for col, mv in missing_values.items():
                    lines.append(f"        - {col}: {mv}")

        if columns:
            lines.append("  Columns:")
            for col_name, col_stats in list(columns.items()):
                lines.append(f"    - {col_name}:")
                for key in ("min_value", "max_value", "mean_value", "std_value", "nb_unique_values", "missing_values"):
                    if key in col_stats and col_stats[key] is not None:
                        lines.append(f"        - {key}: {col_stats[key]}")  # noqa: PERF401

                unique_vals = col_stats.get("unique_values")
                if unique_vals:
                    lines.append(f"        - unique_values_sample: {unique_vals}")

        return "\n".join(lines)

    def format(self) -> str:
        if self.doc_type == DocumentType.QUERY and self.context:
            # Include similarity score for query results
            formatted = "\n\n".join(
                SIMILARITY_FORMAT_TEMPLATE.format(
                    score=f"{result.score:.4f}",
                    content=result.document.metadata.get("full_content", result.document.page_content),
                )
                for result in self.context
            )
        else:
            parts = []

            if self.acronyms:
                parts.append(self.acronyms)

            for result in self.context:
                doc = result.document
                meta = doc.metadata or {}
                content = meta.get("full_content", doc.page_content)

                section_lines = [content]

                stats = meta.get("table_statistics", {})
                if stats:
                    section_lines.append(self._format_table_statistics(stats))

                parts.append("\n\n".join(section_lines))

            formatted = DOCUMENT_SEPARATOR.join(parts)

        return formatted


class CollectionResponse(BaseModel):
    collection: DocumentType
    action: Literal["created", "loaded", "deleted", "recreated"]

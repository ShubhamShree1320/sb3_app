import logging

from langchain_core.documents import Document

from sb3_api.enums.document import DocumentType
from sb3_api.processing.document.base import DocumentProcessor
from sb3_api.profiling.profiler import Profiler

logger = logging.getLogger(__name__)


class TableDocumentProcessor(DocumentProcessor):
    def __init__(self, profiler: Profiler) -> None:
        self._profiler = profiler

    @property
    def get_document_type(self) -> DocumentType:
        return DocumentType.TABLE

    def process_documents(self, documents: list[Document]) -> list[Document]:
        transformed_docs = []

        for doc in documents:
            content = doc.page_content

            fields_start = content.find("Fields:")
            header_and_description = content[:fields_start].strip() if fields_start != -1 else content.strip()
            table_name = self.extract_table_name(content)

            new_metadata = doc.metadata.copy()
            new_metadata["full_content"] = content.strip()
            new_metadata["document_type"] = self.get_document_type.value

            if table_name:
                try:
                    stats = self._profiler.compute_table_statistics(table_name)
                    new_metadata["table_statistics"] = stats
                except Exception:
                    logger.exception("Failed to compute statistics for table %s", table_name)
            transformed_docs.append(Document(page_content=header_and_description, metadata=new_metadata))
        return transformed_docs

    def extract_table_name(self, content: str) -> str | None:
        for line in content.splitlines():
            if line.strip().startswith("Table:"):
                return line.split("Table:")[1].strip()
        return None

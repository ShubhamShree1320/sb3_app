from abc import ABC, abstractmethod

from langchain_core.documents import Document

from sb3_api.enums.document import DocumentType


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""

    @abstractmethod
    def process_documents(self, documents: list[Document]) -> list[Document]:
        """Process documents and return transformed documents."""

    @property
    @abstractmethod
    def get_document_type(self) -> DocumentType:
        """Get the document type this processor handles."""

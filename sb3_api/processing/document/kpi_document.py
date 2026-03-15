from langchain_core.documents import Document

from sb3_api.enums.document import DocumentType
from sb3_api.processing.document.base import DocumentProcessor


class KPIDocumentProcessor(DocumentProcessor):
    @property
    def get_document_type(self) -> DocumentType:
        return DocumentType.KPI

    def process_documents(self, documents: list[Document]) -> list[Document]:
        transformed_docs = []

        for doc in documents:
            content = doc.page_content
            description_start = content.find("Description")
            description_end = content.find("Product Coverage")

            if description_start != -1 and description_end != -1:
                header_and_description = content[description_start:description_end].strip()
            else:
                header_and_description = content.strip()

            new_metadata = doc.metadata.copy()
            new_metadata["full_content"] = content
            new_metadata["document_type"] = self.get_document_type.value

            transformed_docs.append(Document(page_content=header_and_description, metadata=new_metadata))
        return transformed_docs

import logging

from langchain_core.documents import Document

from sb3_api.enums.document import DocumentType
from sb3_api.processing.document.base import DocumentProcessor

logger = logging.getLogger(__name__)

DOCUMENT_TEMPLATE = "Question: {question}\n\nSQL Query:\n{sql}"


class QueryDocumentProcessor(DocumentProcessor):
    @property
    def get_document_type(self) -> DocumentType:
        return DocumentType.QUERY

    def query_pair_validation(self, pair: dict, file: str) -> tuple[bool, str, str]:
        # Validate is a dict
        if not isinstance(pair, dict):
            logger.warning("Skipping non-dict entry in %s", file)
            return False, "", ""

        question = pair.get("question", "").strip()
        sql = pair.get("sql", "").strip()

        # Skip invalid entries
        if not question or not sql:
            logger.warning(
                "Skipping invalid question/sql pair in %s: question=%s, sql=%s", file, bool(question), bool(sql)
            )
            return False, "", ""

        return True, question, sql

    def process_documents(self, documents: list[Document]) -> list[Document]:
        result = []
        for doc in documents:
            query_pairs = doc.metadata.get("query_pairs", [])
            for indx, pair in enumerate(query_pairs):  # Iterate over list of question-sql pairs
                valid, question, sql = self.query_pair_validation(pair, doc.metadata.get("source", ""))

                if valid:
                    source_name = doc.metadata.get("source", "").split(".")[0]
                    source_type = (
                        doc.metadata.get("source", "").split(".")[1]
                        if len(doc.metadata.get("source", "").split(".")) > 1
                        else ""
                    )
                    source = f"{source_name}_{indx}.{source_type}" if source_type else source_name
                    result.append(
                        Document(
                            page_content=question,
                            metadata={
                                "source": source,  # So that each document is unique
                                "question": question,  # Use validated question
                                "sql": sql,  # Use validated sql
                                "full_content": DOCUMENT_TEMPLATE.format(question=question, sql=sql),
                                "document_type": self.get_document_type.value,
                            },
                        )
                    )

        return result

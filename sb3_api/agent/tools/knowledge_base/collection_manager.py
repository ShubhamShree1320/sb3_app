import logging
import time

from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import column, exists, select, table, text
from sqlalchemy.engine import Engine

from sb3_api.encoder import BedrockEncoder
from sb3_api.enums.document import DocumentType
from sb3_api.exceptions.exceptions import CollectionNotFoundError
from sb3_api.loaders.s3_loader import S3Loader
from sb3_api.processing.document.kpi_document import KPIDocumentProcessor
from sb3_api.processing.document.query_document import QueryDocumentProcessor
from sb3_api.processing.document.table_document import TableDocumentProcessor
from sb3_api.profiling.profiler import Profiler
from sb3_api.settings import ServiceSettings
from sb3_api.utils.utils import validate_sql_identifier

logger = logging.getLogger(__name__)


class CollectionManager:
    def __init__(
        self,
        engine: Engine,
        embeddings: BedrockEncoder,
        settings: ServiceSettings,
        profiler: Profiler,
    ) -> None:
        self.engine = engine
        self._embeddings = embeddings
        self.collection_name = settings.COLLECTION_NAME
        self.distance_strategy = settings.DISTANCE_STRATEGY
        self.collection_table_name = validate_sql_identifier(settings.COLLECTION_TABLE_NAME)
        self.embedding_table_name = validate_sql_identifier(settings.EMBEDDING_TABLE_NAME)

        self.loader = S3Loader(settings=settings)
        self.processors = {
            DocumentType.KPI: KPIDocumentProcessor(),
            DocumentType.TABLE: TableDocumentProcessor(profiler=profiler),
            DocumentType.QUERY: QueryDocumentProcessor(),
        }

        self._vector_stores: dict[DocumentType, PGVector] = {}

    def get_collection_name(self, doc_type: DocumentType) -> str:
        return f"{self.collection_name}_{doc_type.value}_description"

    def get_vector_store(self, doc_type: DocumentType) -> PGVector:
        if doc_type not in self._vector_stores:
            collection_name = self.get_collection_name(doc_type)
            if not self.ensure_collection(doc_type):
                logger.info("Collection %s not found", collection_name)
                raise CollectionNotFoundError(collection_name)

            logger.info("Loading collection %s from database", collection_name)
            self._vector_stores[doc_type] = PGVector.from_existing_index(
                embedding=self._embeddings.model,
                collection_name=collection_name,
                distance_strategy=self.distance_strategy,
                connection=self.engine,
                pre_delete_collection=False,
            )
        return self._vector_stores[doc_type]

    def ensure_collection(self, doc_type: DocumentType) -> bool:
        collection_name = self.get_collection_name(doc_type)
        with self.engine.connect() as conn:
            # Check if collection table exists
            table_exists = conn.execute(
                text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = :table_name
                    )
                """),
                {"table_name": self.collection_table_name},
            ).scalar()

            if not table_exists:
                logger.info("Table %s does not exist", self.collection_table_name)
                return False

            # Get collection ID
            collection_id = conn.execute(
                select(column("uuid"))
                .select_from(table(self.collection_table_name))
                .where(column("name") == collection_name)
            ).scalar()

            if collection_id is None:
                logger.info("Collection %s not found", collection_name)
                return False

            # Check for embeddings
            has_embeddings = conn.execute(
                select(
                    exists(
                        select(1)
                        .select_from(table(self.embedding_table_name))
                        .where(column("collection_id") == collection_id)
                    )
                )
            ).scalar()

            return bool(has_embeddings)

    def load_documents(self, doc_type: DocumentType) -> list[Document]:
        prefix = self.loader.prefix_map[doc_type]
        return self.loader.load_documents_from_s3(prefix)

    def process_documents(self, doc_type: DocumentType) -> list[Document]:
        documents = self.load_documents(doc_type)
        processor = self.processors[doc_type]
        return processor.process_documents(documents)

    # def create_or_load_collection(self, doc_type: DocumentType) -> None:
    #     collection_name = self.get_collection_name(doc_type)

    #     try:
    #         self.get_vector_store(doc_type)
    #         logger.info("Collection %s already exists", collection_name)

    #     except CollectionNotFoundError:
    #         logger.info("Creating collection %s", collection_name)
    #         processed_docs = self.process_documents(doc_type)
    #         vector_store = PGVector.from_documents(
    #             processed_docs,
    #             self._embeddings.model,
    #             collection_name=collection_name,
    #             distance_strategy=self.distance_strategy,
    #             connection=self.engine,
    #             pre_delete_collection=False,
    #         )
    #         self._vector_stores[doc_type] = vector_store

    def create_or_load_collection(self, doc_type: DocumentType) -> None:
        collection_name = self.get_collection_name(doc_type)

        try:
            self.get_vector_store(doc_type)
            logger.info("Collection %s already exists", collection_name)

        except CollectionNotFoundError:
            logger.info("Creating collection %s", collection_name)
            processed_docs = self.process_documents(doc_type)
            
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    vector_store = PGVector.from_documents(
                        processed_docs,
                        self._embeddings.model,
                        collection_name=collection_name,
                        distance_strategy=self.distance_strategy,
                        connection=self.engine,
                        pre_delete_collection=False,
                    )
                    self._vector_stores[doc_type] = vector_store
                    break
                except Exception as e:
                    if "ThrottlingException" in str(e) and attempt < max_retries - 1:
                        wait_time = 30 * (attempt + 1)
                        logger.warning("Throttled by Bedrock, waiting %s seconds before retry %s/%s", wait_time, attempt + 1, max_retries)
                        time.sleep(wait_time)
                    else:
                        raise

    def create_all_collections(self) -> None:
        for doc_type in DocumentType:
            self.create_or_load_collection(doc_type)

    def delete_collection(self, doc_type: DocumentType) -> None:
        collection_name = self.get_collection_name(doc_type)
        try:
            vector_store = self.get_vector_store(doc_type)
            vector_store.delete_collection()
            self._vector_stores.pop(doc_type, None)
        except CollectionNotFoundError:
            logger.info("Collection %s not found", collection_name)
            raise

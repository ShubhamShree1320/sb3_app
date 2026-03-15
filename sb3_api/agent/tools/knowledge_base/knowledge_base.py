import logging

from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from sb3_api.agent.tools.knowledge_base.collection_manager import CollectionManager
from sb3_api.agent.tools.knowledge_base.search_engine import SearchEngine
from sb3_api.encoder import BedrockEncoder
from sb3_api.enums.document import DocumentType
from sb3_api.loaders.s3_loader import S3Loader
from sb3_api.models.knowledge_base import DocumentContext
from sb3_api.profiling.profiler import Profiler
from sb3_api.settings import ServiceSettings

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(
        self,
        settings: ServiceSettings | None = None,
        url: str | None = None,
        embeddings: BedrockEncoder | None = None,
        profiler: Profiler | None = None,
    ) -> None:
        if settings is None or url is None or embeddings is None or profiler is None:
            raise ValueError("First instantiation requires all parameters")

        self.settings = settings

        self.loader = S3Loader(settings=settings)
        self._acronyms = self.loader.read_s3_text_file(self.loader.acronyms_key)
        self.engine = create_engine(
            url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )

        loader = S3Loader(settings=settings)
        self._acronyms = loader.read_s3_text_file(loader.acronyms_key)

        self.collection_manager = CollectionManager(
            engine=self.engine,
            embeddings=embeddings,
            settings=settings,
            profiler=profiler,
        )

        self.search_engine = SearchEngine(
            engine=self.engine, collection_manager=self.collection_manager, settings=settings
        )

        self.search_type = self.settings.SEARCH_TYPE

        self.query_search_results_k = self.settings.QUERY_SEARCH_RESULTS_K
        self.context_search_results_k = self.settings.CONTEXT_SEARCH_RESULTS_K

    def ensure_collection(self, doc_type: DocumentType) -> bool:
        return self.collection_manager.ensure_collection(doc_type)

    def create_or_load_collection(self, doc_type: DocumentType) -> None:
        self.collection_manager.create_or_load_collection(doc_type)

    def delete_collection(self, doc_type: DocumentType) -> None:
        self.collection_manager.delete_collection(doc_type)

    def create_all_collections(self) -> None:
        self.collection_manager.create_all_collections()

    def retrieve_context(
        self,
        query: str,
        doc_type: DocumentType,
    ) -> DocumentContext:
        results = DocumentContext(context=[], doc_type=doc_type)
        try:
            k = self.query_search_results_k if doc_type == DocumentType.QUERY else self.context_search_results_k
            results.context = self.search_engine.search(
                search_type=self.search_type, query=query, doc_type=doc_type, k=k
            )
            results.acronyms = self._acronyms
        except Exception:
            logger.exception("Unexpected error retrieving context for %s", doc_type)
        return results

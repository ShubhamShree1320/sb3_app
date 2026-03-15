import logging
from functools import partial
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from sqlalchemy import column, select, table, text
from sqlalchemy.engine import Engine

from sb3_api.agent.tools.knowledge_base.collection_manager import CollectionManager
from sb3_api.enums.document import DocumentType
from sb3_api.enums.search import SearchType
from sb3_api.models.knowledge_base import SearchResult
from sb3_api.settings import ServiceSettings
from sb3_api.utils.utils import validate_sql_identifier

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, engine: Engine, collection_manager: CollectionManager, settings: ServiceSettings) -> None:
        self.engine = engine
        self.collection_manager = collection_manager
        self._search_functions: dict[SearchType, Callable] = {
            SearchType.SEMANTIC: self.semantic_search,
            SearchType.KEYWORD: self.keyword_search,
            SearchType.HYBRID: partial(
                self.hybrid_search,
                weight=settings.HYBRID_SEMANTIC_WEIGHT,
            ),
        }
        self.embedding_table_name = validate_sql_identifier(settings.EMBEDDING_TABLE_NAME)
        self.collection_table_name = validate_sql_identifier(settings.COLLECTION_TABLE_NAME)

        self.cos_distance_threshold_query = settings.COS_DISTANCE_THRESHOLD_QUERY
        self.cos_distance_threshold_context = settings.COS_DISTANCE_THRESHOLD_CONTEXT

        # use plainto_tsquery to split the user query in keywords
        # we also modify the AND statements with OR statements because by default
        # PostgreSQL uses AND for all conditions,
        # and we want to match documents that contain any of the keywords (OR) not all keywords (AND)
        # coalesce is used to handle the case where we don't have a full_content field in the metadata
        self._keyword_search_query = text(f"""
                WITH search_query AS (
                    SELECT plainto_tsquery('english', :query) as base_query
                ),
                or_query AS (
                    SELECT
                        CASE
                            WHEN base_query::text = '' THEN 'none'::tsquery
                            ELSE replace(base_query::text, ' & ', ' | ')::tsquery
                        END as query
                    FROM search_query
                )
                SELECT
                    document,
                    cmetadata,
                    ts_rank(
                        to_tsvector('english', COALESCE(cmetadata->>'full_content', document)),
                        or_query.query
                    ) as rank
                FROM "{self.embedding_table_name}", or_query
                WHERE
                    collection_id = :collection_id
                    AND to_tsvector('english', COALESCE(cmetadata->>'full_content', document))
                        @@ or_query.query
                ORDER BY rank DESC
                LIMIT :k
            """)  # noqa: S608

    # Compute Reciprocal Rank Fusion scores for ranked lists: this combines the scores of different search types
    # The idea is to give higher scores to documents that appear in multiple ranked lists
    # As the semantic search and keyword search uses different scoring mechanisms, we need to account for that
    def _compute_rrf_scores(
        self, ranked_lists: list[list[Document]], weight: float, c: int = 60, id_key: str = ""
    ) -> dict[str, float]:
        scores: dict[str, float] = {}

        weights = [weight, 1 - weight]

        for docs, w in zip(ranked_lists, weights, strict=True):
            for rank, doc in enumerate(docs, start=1):
                key = doc.metadata[id_key] if id_key else doc.page_content
                score = scores.get(key, 0.0)
                scores[key] = score + w / (rank + c)

        return scores

    def semantic_search(self, query: str, doc_type: DocumentType, k: int) -> list[SearchResult]:
        logger.info("Performing semantic search")
        vector_store = self.collection_manager.get_vector_store(doc_type)
        results = vector_store.similarity_search_with_score(query, k=k)

        # Apply document-type-specific threshold
        threshold = (
            self.cos_distance_threshold_query if doc_type == DocumentType.QUERY else self.cos_distance_threshold_context
        )
        filtered_results = [(doc, score) for doc, score in results if score <= threshold]

        if len(results) != len(filtered_results):
            logger.debug(
                "Filtered %d results below threshold %.2f for %s",
                len(results) - len(filtered_results),
                threshold,
                doc_type,
            )

        unique_results = [
            SearchResult(document=doc, score=score)
            for doc, score in {doc.metadata.get("source"): (doc, score) for doc, score in results}.values()
        ]
        return unique_results

    def keyword_search(self, query: str, doc_type: DocumentType, k: int) -> list[SearchResult]:
        logger.info("Performing keyword search")
        collection_name = self.collection_manager.get_collection_name(doc_type)

        with self.engine.connect() as conn:
            # Get collection ID
            collection_id = conn.execute(
                select(column("uuid"))
                .select_from(table(self.collection_table_name))
                .where(column("name") == collection_name)
            ).scalar()

            if collection_id is None:
                logger.warning("Collection %s not found for keyword search", collection_name)
                return []

            results = conn.execute(
                self._keyword_search_query,
                {"query": query, "collection_id": collection_id, "k": k},
            ).fetchall()

        return [
            SearchResult(document=Document(page_content=row.document, metadata=row.cmetadata or {}), score=row.rank)
            for row in results
        ]

    def hybrid_search(self, query: str, doc_type: DocumentType, k: int, weight: float) -> list[SearchResult]:
        logger.info("Performing hybrid search")
        sem_results = self.semantic_search(query, doc_type, k)
        kw_results = self.keyword_search(query, doc_type, k)

        # we use only the ranking of the documents
        sem_docs = [r.document for r in sem_results]
        kw_docs = [r.document for r in kw_results]

        scores = self._compute_rrf_scores([sem_docs, kw_docs], weight=weight, c=60, id_key="source")

        merged = {doc.metadata.get("source", doc.page_content): doc for doc in sem_docs + kw_docs}

        ranked = sorted(
            merged.values(), key=lambda d: scores.get(d.metadata.get("source", d.page_content), 0), reverse=True
        )

        # return top k with RRF scores
        results = []
        for doc in ranked[:k]:
            key = doc.metadata.get("source", doc.page_content)
            results.append(SearchResult(document=doc, score=scores[key]))

        return results

    def search(self, search_type: SearchType, query: str, doc_type: DocumentType, k: int) -> list[SearchResult]:
        search_function = self._search_functions[search_type]
        return search_function(query=query, doc_type=doc_type, k=k)

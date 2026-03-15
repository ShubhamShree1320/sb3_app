from enum import Enum


class SearchType(Enum):
    """Enum for different search strategies in the knowledge base."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"

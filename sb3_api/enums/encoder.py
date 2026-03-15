from enum import Enum

# Embedding API
TITAN_BEDROCK_ID = "amazon.titan-embed-text-v1"
TITAN_EMBEDDINGS_SIZE = 1536
TITAN_MAX_CHARS = 48_000  # max chars: 8k tokens ~= 48k chars according to Amazon (1 token = 6 chars)

COHERE_V3_BEDROCK_ID = "cohere.embed-multilingual-v3"
COHERE_V3_EMBEDDINGS_SIZE = 1024
COHERE_V3_MAX_CHARS = 2024  # max chars from Cohere API

COHERE_V4_BEDROCK_ID = "cohere.embed-v4:0"
COHERE_V4_EMBEDDINGS_SIZE = 1024
COHERE_V4_MAX_CHARS = 128000


class EncoderModel(Enum):
    TITAN_V1 = (
        "TITAN_V1",
        TITAN_BEDROCK_ID,
        TITAN_EMBEDDINGS_SIZE,
        TITAN_MAX_CHARS,
    )
    COHERE_V3 = (
        "COHERE_V3",
        COHERE_V3_BEDROCK_ID,
        COHERE_V3_EMBEDDINGS_SIZE,
        COHERE_V3_MAX_CHARS,
    )
    COHERE_V4 = (
        "COHERE_V4",
        COHERE_V4_BEDROCK_ID,
        COHERE_V4_EMBEDDINGS_SIZE,
        COHERE_V4_MAX_CHARS,
    )

    def __init__(self, name: str, aws_id: str, embeddings_size: int, max_chars: int) -> None:
        self._name = name
        self._aws_id = aws_id
        self._embeddings_size = embeddings_size
        self._max_chars = max_chars

    @property
    def value(self) -> str:  # type: ignore[override]
        return self._name

    @property
    def aws_id(self) -> str:
        """Embeddings Model ID according to Amazon Bedrock API."""
        return self._aws_id

    @property
    def embeddings_size(self) -> int:
        """Number of dimensions for embedding vectors."""
        return self._embeddings_size

    @property
    def max_chars(self) -> int:
        """Maximum number of chars that can be encoded."""
        return self._max_chars

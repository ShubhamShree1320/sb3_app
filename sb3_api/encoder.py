import logging

from botocore.config import Config
from langchain_aws import BedrockEmbeddings

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3


class BedrockEncoder:
    def __init__(
        self,
        model_id: str,
        *,
        normalize: bool = True,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        config = Config(
            retries={"max_attempts": max_retries, "mode": "standard"},
            connect_timeout=timeout,
            read_timeout=timeout,
        )
        self.model = BedrockEmbeddings(model_id=model_id, normalize=normalize, config=config)

    async def aembed_query(self, text: str) -> list[float]:
        """Compute query embeddings using a Bedrock model."""
        embedding = await self.model.aembed_query(text)

        if not embedding:
            raise ValueError("Failed to compute embeddings for the given input text")

        return embedding

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = await self.model.aembed_documents(texts)

        if not embeddings:
            raise ValueError("Failed to compute embeddings for the given input texts")

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.embed_query(text)

        if not embedding:
            raise ValueError("Failed to compute embeddings for the given input text")

        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.embed_documents(texts)

        if not embeddings:
            raise ValueError("Failed to compute embeddings for the given input texts")

        return embeddings

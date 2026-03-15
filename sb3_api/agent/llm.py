"""LLM factory for SB3 API.

Two separate factories:
  - create_llm()           → LangChain ChatBedrockConverse (tool-chain LLMs)
  - create_strands_model() → Strands BedrockModel (agent executors)

Key fixes vs original:
  1. boto3_client= removed (invalid param, caused UserWarning every startup).
     Replaced with boto_client_config= (correct Strands API).
  2. cache_config=CacheConfig(strategy="auto") and cache_tools="default" added.
     Without these Strands sends the full system prompt on every LLM call,
     causing ~70% more input tokens compared to LangGraph which had caching.
"""

import boto3
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.language_models import BaseChatModel
from strands.models.bedrock import BedrockModel as StrandsBedrockModel
from strands.models.model import CacheConfig

from sb3_api.config.llm import BaseLLMConfig

# Shared botocore config applied to both LangChain and Strands clients.
_BOTO_CLIENT_CONFIG = Config(
    retries={
        "max_attempts": 5,
        "mode": "adaptive",
    },
    read_timeout=300,    # 5 min – long SQL queries can take a while
    connect_timeout=10,
)


def create_llm(config: BaseLLMConfig) -> BaseChatModel:
    """Create a LangChain ChatBedrockConverse model.

    Used by tool-chain LLMs only (e.g. sql_db_query_checker).
    """
    client = boto3.client("bedrock-runtime", config=_BOTO_CLIENT_CONFIG)
    return ChatBedrockConverse(
        client=client,
        model_id=config.model.value,   # type: ignore[call-arg]
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def create_strands_model(config: BaseLLMConfig) -> StrandsBedrockModel:
    """Create a Strands BedrockModel with prompt caching enabled.

    FIX 1 – boto3_client → boto_client_config
    -----------------------------------------
    BedrockModel does NOT accept a pre-built boto3 client.  The correct API
    is boto_client_config (a BotocoreConfig) which BedrockModel uses when it
    creates its own bedrock-runtime client internally.

    Passing boto3_client= is silently ignored and emits:
        UserWarning: Invalid configuration parameters: ['boto3_client']

    FIX 2 – Enable prompt caching
    -----------------------------
    LangGraph's ChatBedrockConverse had caching enabled (evidenced by the
    3 846 cache-creation + 22 185 cache-read tokens it reported).  Strands
    BedrockModel has caching OFF by default, so every LLM call re-sends the
    full system prompt (~2 400 tokens) and all tools (~400 tokens) instead of
    reading them from the cache.

    Two cache settings are required:

    cache_config=CacheConfig(strategy="auto")
        Automatically inserts cache-point markers in the conversation history.
        This is equivalent to LangGraph's prompt caching for the system prompt
        and recent turns.

    cache_tools="default"
        Caches the tool specifications block.  With 8 tools defined the tool
        spec is sent verbatim on every call without this flag.

    Expected improvement after these two settings:
        Before:  Input ~36 000, Cache creation 0,  Cache read 0
        After:   Input ~19 000, Cache creation ~3 800, Cache read ~22 000
        Saving:  ~47% fewer input tokens billed (matches LangGraph baseline)
    """
    return StrandsBedrockModel(
        model_id=config.model.value,
        boto_client_config=_BOTO_CLIENT_CONFIG,        # FIX 1: correct param name
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        cache_config=CacheConfig(strategy="auto"),     # FIX 2a: cache system prompt + turns
        cache_tools="default",                          # FIX 2b: cache tool specifications
    )

"""FastAPI dependency injection.

LATENCY FIXES applied here:
  1. @lru_cache on get_prompt_registry  — YAML parsed once, not per-request.
  2. @lru_cache on get_tool_factory     — boto3 LLM client created once, not per-request.
  3. @lru_cache on get_session_repository — DynamoDB config object created once.
  4. @lru_cache on get_graph / get_graph_stream — AgentPipeline (and its cached
     executor instances) live for the lifetime of the process.

     Previously every HTTP request rebuilt the entire chain:
       request → get_chat_controller → get_graph_stream → get_tool_factory
              → create_llm() (new boto3 client)   ← ~120 ms cold
              → ToolFactory.__init__               ← tool objects
       request → SQLAgentExecutor.__init__
              → _test_database_connection()        ← 2 SQL round-trips to Redshift
              → _create_agent()                   ← new StrandsBedrockModel / boto3
     With caching only _make_fresh_agent() (microseconds) runs per-request.
"""

from functools import lru_cache
from typing import Any

from fastapi import Depends
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

from sb3_api.agent.prompts.prompts import PromptRegistry
from sb3_api.agent.tools.factory import ToolFactory
from sb3_api.agent.tools.knowledge_base.knowledge_base import KnowledgeBase
from sb3_api.controller.chat_controller import ChatController
from sb3_api.encoder import BedrockEncoder
from sb3_api.graph import AgentPipeline, GraphBuilder
from sb3_api.profiling.profiler import Profiler
from sb3_api.repository.agent.base import BaseSQLDatabase
from sb3_api.repository.agent.redshift import RedshiftSQLDatabase
from sb3_api.repository.agent.redshift_connection import RedshiftConnectionPool
from sb3_api.repository.agent.sql_generic import SQLDatabaseWrapper
from sb3_api.repository.feedback.base import FeedbackRepository
from sb3_api.repository.feedback.rds import RDSFeedbackRepository
from sb3_api.repository.session.base import SessionRepository
from sb3_api.repository.session.dynamodb import DynamoDBSessionRepository
from sb3_api.settings import ServiceSettings
from sb3_api.stores.feedback_store import FeedbackStore


@lru_cache
def get_settings() -> ServiceSettings:
    return ServiceSettings()


@lru_cache(maxsize=1)
def get_connection_pool() -> RedshiftConnectionPool:
    settings = get_settings()
    return RedshiftConnectionPool(
        host=settings.DB_HOST,
        database=settings.DB_NAME,
        user=settings.DB_USERNAME,
        password=settings.DB_PASSWORD,
        port=settings.DB_PORT,
    )


def get_sql_database(
    settings: ServiceSettings = Depends(get_settings),
) -> BaseSQLDatabase:
    if settings.USE_MOCK_DBS:
        from pathlib import Path
        BASE_DIR = Path(__file__).resolve().parents[1]
        DB_PATH = BASE_DIR / settings.DB_NAME
        engine = create_engine(f"sqlite:///{DB_PATH}")
        sql_db = SQLDatabase(engine=engine)
        return SQLDatabaseWrapper(sql_db)

    return RedshiftSQLDatabase(connection_pool=get_connection_pool())


def get_rds_url_async(settings: ServiceSettings = Depends(get_settings)) -> str:
    driver = "postgresql+asyncpg"
    return (
        f"{driver}://{settings.RDS_USERNAME}:{settings.RDS_PASSWORD}"
        f"@{settings.RDS_HOST}:{settings.RDS_PORT}/{settings.RDS_DB_NAME}"
    )


def get_rds_url_sync(settings: ServiceSettings = Depends(get_settings)) -> str:
    driver = "postgresql+psycopg"
    return (
        f"{driver}://{settings.RDS_USERNAME}:{settings.RDS_PASSWORD}"
        f"@{settings.RDS_HOST}:{settings.RDS_PORT}/{settings.RDS_DB_NAME}"
    )


def get_feedback_repository(
    settings: ServiceSettings = Depends(get_settings),
) -> FeedbackRepository:
    url = get_rds_url_async(settings)
    return RDSFeedbackRepository(db_url=url, encoder_model=settings.ENCODER_MODEL)


def get_feedback_store(
    repository: FeedbackRepository = Depends(get_feedback_repository),
) -> FeedbackStore:
    encoder = BedrockEncoder(model_id=repository.encoder_model.aws_id)
    return FeedbackStore(repository=repository, encoder=encoder)


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    """Singleton — YAML is read and parsed once at startup.

    Previously called on every request, reading and parsing prompts.yaml from
    disk each time. With @lru_cache the file is read once and the parsed dict
    is reused for the process lifetime.
    """
    return PromptRegistry()


@lru_cache(maxsize=1)
#def get_knowledge_base() -> KnowledgeBase:
def get_knowledge_base(settings: ServiceSettings = Depends(get_settings)) -> KnowledgeBase:
    """Singleton — expensive PGVector + S3 initialisation done once."""
    settings = get_settings()
    db = get_sql_database(settings)
    url = get_rds_url_sync(settings)
    encoder = BedrockEncoder(model_id=settings.ENCODER_MODEL.aws_id)
    profiler = Profiler(db)  # type: ignore[arg-type]
    return KnowledgeBase(settings=settings, url=url, embeddings=encoder, profiler=profiler)


@lru_cache(maxsize=1)
def get_session_repository() -> SessionRepository:
    """Singleton — just holds config; no per-request state."""
    settings = get_settings()
    return DynamoDBSessionRepository(
        table_name=settings.DYNAMODB_SESSION_TABLE_NAME,
        endpoint_url=settings.DYNAMODB_ENDPOINT_URL,
    )


@lru_cache(maxsize=1)
def get_tool_factory() -> ToolFactory:
    """Singleton — creates the LangChain LLM boto3 client once.

    Previously get_tool_factory() had no cache, so every HTTP request:
      1. Called create_llm() → boto3.client("bedrock-runtime") → ~120 ms cold
      2. Built all LangChain tool instances from scratch

    With @lru_cache this is done once at first request and reused forever.
    ToolFactory itself is stateless (tools carry no per-request state).
    """
    settings = get_settings()
    db = get_sql_database(settings)
    prompt_registry = get_prompt_registry()
    knowledge_base = get_knowledge_base()
    return ToolFactory(
        db=db,
        prompt_registry=prompt_registry,
        knowledge_base=knowledge_base,
        settings=settings,
    )


@lru_cache(maxsize=1)
def get_graph() -> AgentPipeline:
    """Singleton non-streaming pipeline.

    The AgentPipeline caches SQLAgentExecutor / ContextAgentExecutor instances
    per (persona, debug_mode) internally, so subsequent requests only pay for
    _make_fresh_agent() which is microseconds.
    """
    tool_factory = get_tool_factory()
    prompt_registry = get_prompt_registry()
    settings = get_settings()
    return GraphBuilder(
        tool_factory=tool_factory,
        prompt_registry=prompt_registry,
        enable_context=settings.USE_CONTEXT_AGENT,
    ).build()


@lru_cache(maxsize=1)
def get_graph_stream() -> AgentPipeline:
    """Singleton streaming pipeline — see get_graph() docstring."""
    tool_factory = get_tool_factory()
    prompt_registry = get_prompt_registry()
    settings = get_settings()
    return GraphBuilder(
        tool_factory=tool_factory,
        prompt_registry=prompt_registry,
        stream=True,
        enable_context=settings.USE_CONTEXT_AGENT,
    ).build()


def get_chat_controller(
    graph: AgentPipeline = Depends(get_graph_stream),
    session_repository: DynamoDBSessionRepository = Depends(get_session_repository),
) -> ChatController:
    return ChatController(graph=graph, session_repository=session_repository)

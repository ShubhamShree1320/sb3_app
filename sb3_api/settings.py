from langchain_postgres.vectorstores import DistanceStrategy
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sb3_api.enums.encoder import EncoderModel
from sb3_api.enums.environment import Environment
from sb3_api.enums.search import SearchType


class SQLDatabaseSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file="local.env",
        env_file_encoding="utf-8",
    )

    DB_USERNAME: str = ""
    """DB username."""

    DB_PASSWORD: str = ""
    """DB password."""

    DB_HOST: str = "localhost"
    """DB host."""

    DB_PORT: int = 5439
    """DB port."""

    DB_NAME: str = "new_sb3.db"
    """DB name."""

    DB_POOL_SIZE: int = 4
    """DB connection pool size."""

    DB_RESET_AFTER: int = 50
    """DB connection reset after N queries."""

    DB_MAX_CONNECTION_AGE_MINUTES: int = 30
    """DB connection max age in minutes."""


class RDSSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="local.env",
        env_file_encoding="utf-8",
    )
    RDS_USERNAME: str = "admin"
    """DB username."""

    RDS_PASSWORD: str = "dummy"  # noqa: S105
    """DB password."""

    RDS_HOST: str = "localhost"
    """DB host."""

    RDS_PORT: int = 5432
    """DB port."""

    RDS_DB_NAME: str = "dummy"
    """DB name."""


class DynamoDBSettings(BaseSettings):
    DYNAMODB_SESSION_TABLE_NAME: str = "chat-sessions"
    """DynamoDB table name for sessions."""

    DYNAMODB_SESSION_GSI_NAME: str = "user-sessions"
    """Global Secondary Index name for sessions."""

    DYNAMODB_ENDPOINT_URL: str | None = None
    """DynamoDB endpoint URL (for local)."""


class PGVectorSettings(BaseSettings):
    COLLECTION_NAME: str = "kb-collection"

    DISTANCE_STRATEGY: DistanceStrategy = DistanceStrategy.COSINE

    COLLECTION_TABLE_NAME: str = "langchain_pg_collection"

    EMBEDDING_TABLE_NAME: str = "langchain_pg_embedding"


class SearchSettings(BaseSettings):
    SEARCH_TYPE: SearchType = SearchType.SEMANTIC
    """Type of search to use: semantic, keyword, or hybrid."""

    HYBRID_SEMANTIC_WEIGHT: float = Field(default=0.5, ge=0, le=1)
    """Weight for semantic search in hybrid mode (0-1)."""

    CONTEXT_SEARCH_RESULTS_K: int = Field(default=3, ge=1)
    """Number of search results to return for context search."""

    QUERY_SEARCH_RESULTS_K: int = Field(default=1, ge=1)
    """Number of search results to return for query search."""

    COS_DISTANCE_THRESHOLD_QUERY: float = Field(default=0.5, ge=0, le=2)
    """Maximum cosine distance for query search (0=identical, 2=opposite). Stricter threshold for precise matching."""

    COS_DISTANCE_THRESHOLD_CONTEXT: float = Field(default=0.9, ge=0, le=2)
    """Maximum cosine distance for KPI/table search. More permissive to capture related business context."""


class AuthSettings(BaseSettings):
    AUTH_CLIENT_ID: str = ""

    AUTH_CLIENT_SECRET: str = ""

    AUTH_AUTHORIZATION_URL: str = "https://sso-corproot-v2.scapp-services.swisscom.com/oauth/authorize"

    AUTH_TOKEN_URL: str = "https://sso-corproot-v2.scapp-services.swisscom.com/oauth/token"  # noqa: S105

    AUTH_USER_INFO_URL: str = "https://sso-corproot-v2.scapp-services.swisscom.com/userinfo"

    AUTH_REDIRECT_URI: str = ""

    AUTH_DISABLED: bool = False
    """Flag to toggle authentication."""

    AUTH_ALLOWED_USERS: str = ""
    """Comma separated string of email addresses."""

    AUTH_ADMIN_USERS: str = ""
    """Comma separated string of email addresses for admin users."""

    AUTH_AUDIENCE: str = ""
    """OAuth2 audience claim for JWT token validation; same as AUTH_CLIENT_ID."""


class S3Settings(BaseSettings):
    S3_BUCKET: str = "sb3-tv-sales-metadata"

    KPI_PREFIX: str = "kpis/"

    TABLE_PREFIX: str = "tables/"

    QUERY_PREFIX: str = "queries-examples/"

    ACRONYMS_KEY: str = "acronyms.txt"


class ServiceSettings(
    AuthSettings,
    RDSSettings,
    SQLDatabaseSettings,
    DynamoDBSettings,
    S3Settings,
    PGVectorSettings,
    SearchSettings,
):

    model_config = SettingsConfigDict(
        env_file="local.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    SERVICE_PORT: int = 8000
    """Port number on which the service will run."""

    SERVICE_NAME: str = "sb3-api"
    """Name of the service, displayed in Swagger as app title."""

    API_PREFIX: str = "/api"
    """API prefix for all routes."""

    ENV: Environment = Environment.DEV
    """Environment in which the service is running."""

    ENCODER_MODEL: EncoderModel = EncoderModel.COHERE_V3

    USE_MOCK_DBS: bool = True
    """Use SQLite for local development instead of PostgreSQL/Redshift."""

    USE_CONTEXT_AGENT: bool = False
    """Use context agent for enhanced conversation understanding."""

    def __hash__(self) -> int:
        return hash((type(self), *tuple(self.__dict__.values())))

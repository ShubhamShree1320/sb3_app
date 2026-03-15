"""DynamoDB session repository.

LATENCY FIX:
  _get_client() previously called boto3.resource() on every single operation
  (get_item, put_item, query). Each call creates a new botocore session,
  which involves parsing config files and creating an HTTP connection pool.

  Fix: cache the boto3 ServiceResource at instance level.
  The resource object is thread-safe and holds a connection pool internally,
  so reuse is safe and eliminates repeated setup overhead (~5-15 ms per call).

TOKEN OPTIMISATION:
  messages_to_dict / messages_from_dict (LangChain serialisation) store the
  full LangChain type envelope. This is fine for correctness; if DynamoDB item
  size becomes a concern, consider storing only role + content text for
  user/assistant turns (the only turns the agent reads back) and dropping
  tool_use / tool_result blocks which are never re-injected.
"""

from typing import TYPE_CHECKING, Any
from uuid import UUID

import boto3
from boto3.dynamodb.conditions import Key
from langchain_core.messages.base import messages_to_dict
from langchain_core.messages.utils import messages_from_dict

from sb3_api.exceptions.exceptions import AuthorizationError, SessionNotFoundError, UserNotFoundError
from sb3_api.models.session import SessionChat
from sb3_api.models.session_dto import SessionInfoDTO
from sb3_api.repository.session.base import SessionRepository

if TYPE_CHECKING:
    from mypy_boto3_dynamodb.service_resource import DynamoDBServiceResource, Table


class DynamoDBSessionRepository(SessionRepository):
    PARTITION_KEY: str = "session_id"
    GSI_NAME: str = "user-sessions"
    GSI_PARTITION_KEY: str = "user"

    def __init__(
        self,
        table_name: str,
        endpoint_url: str | None = None,
        region_name: str = "eu-central-1",
    ) -> None:
        self.table_name = table_name
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        # Cache the boto3 resource so we don't re-create it on every operation.
        # boto3.resource is thread-safe and manages its own connection pool.
        self._client: "DynamoDBServiceResource | None" = None

    def _get_client(self) -> "DynamoDBServiceResource":
        """Return the cached boto3 DynamoDB resource, creating it on first call."""
        if self._client is None:
            self._client = boto3.resource(
                service_name="dynamodb",
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
                # NOTE: remove the dummy credentials below in production.
                # They are only here for local DynamoDB testing.
                aws_access_key_id="dummy",
                aws_secret_access_key="dummy",
            )
        return self._client

    def _get_table(self) -> "Table":
        """Return the DynamoDB Table object (also cached implicitly by boto3)."""
        return self._get_client().Table(self.table_name)

    def get_session(self, session_id: UUID, user: str) -> SessionChat:
        key: dict[str, str] = {self.PARTITION_KEY: str(session_id)}
        item = self._get_item(key)

        if not item:
            raise SessionNotFoundError(session_id)

        if item["user"] != user:
            raise AuthorizationError(user)

        content = messages_from_dict(item.get("content", []))
        item["content"] = content
        return SessionChat.model_validate(item)

    def get_sessions_by_user(self, user: str) -> list[SessionInfoDTO]:
        key_condition = Key(self.GSI_PARTITION_KEY).eq(user)
        items = self._query(key_condition, index_name=self.GSI_NAME)

        if not items:
            raise UserNotFoundError(user)

        return [SessionInfoDTO.from_dynamo_item(item) for item in reversed(items)]

    def _get_item(self, key: dict[str, str]) -> dict[str, Any]:
        table = self._get_table()
        response = table.get_item(Key=key)
        return response.get("Item", {})

    def save_session(self, session: SessionChat) -> None:
        """Persist session to DynamoDB.

        TOKEN OPTIMISATION NOTE:
          session.content holds the full LangChain message list (HumanMessage +
          AIMessage only, since the controller only appends those). Tool-use and
          tool-result blocks are NOT stored here, which keeps DynamoDB items small
          and prevents context explosion on multi-turn sessions.
          If you need further savings, store only the last N turns:
              content_to_store = session.content[-MAX_STORED_TURNS:]
        """
        table = self._get_table()
        item_dict = session.model_dump(by_alias=True)
        item_dict["content"] = messages_to_dict(session.content)
        table.put_item(Item=item_dict)

    def _query(self, key_condition: Any, index_name: str | None = None) -> list[dict]:
        table = self._get_table()
        query_kwargs = {"KeyConditionExpression": key_condition}
        if index_name:
            query_kwargs["IndexName"] = index_name
        response = table.query(**query_kwargs)
        return response.get("Items", [])

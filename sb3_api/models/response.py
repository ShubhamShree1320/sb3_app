from langchain_core.messages import BaseMessage
from langchain_core.messages.tool import ToolCall
from pydantic import BaseModel


class Trace(BaseModel):
    content: str
    metadata: dict = {}
    actions: list[ToolCall] = []
    type: str


class AgentStreamResponse(BaseModel):
    trace: Trace
    session_id: str | None = None


class AgentResponse(BaseModel):
    content: str = ""
    messages: list[BaseMessage] = []
    traces: list[Trace] | None = None
    session_id: str | None = None
    plot: str | None = None
    sql_query: str | None = None
    is_clarification: bool = False


class SQLResponse(BaseModel):
    query: str
    content: list[dict] | str | object


class UserInfo(BaseModel):
    user_name: str
    given_name: str
    family_name: str
    email: str
    department: str


class AuthorizeResponse(BaseModel):
    authorization_url: str
    client_id: str
    redirect_uri: str
    state: str


class CheckTokenResponse(BaseModel):
    valid: bool
    user_info: UserInfo


class ExchangeTokenResponse(BaseModel):
    access_token: str
    expires_in: int
    user_info: UserInfo | None

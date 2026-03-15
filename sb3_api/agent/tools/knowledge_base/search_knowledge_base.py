from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from sb3_api.agent.tools.knowledge_base.knowledge_base import KnowledgeBase
from sb3_api.enums.document import DocumentType
from sb3_api.enums.trace import TraceType


class KnowledgeBaseSearchInput(BaseModel):
    """Input for the knowledge base search tool."""

    query: str = Field(
        description="The search query. For primary searches,"
        "use the user's complete original question exactly as asked.",
        min_length=1,
    )
    context_type: Literal["table", "kpi", "query"] = Field(description=("The type of context to retrieve."))


class KnowledgeBaseSearchTool(BaseTool):
    """Tool for searching the knowledge base to get additional context about tables, KPIs, and business terminology."""

    name: str = "knowledge_base_search"
    description: str = """
    Search the knowledge base for comprehensive business context about tables, KPIs, and past query examples.

    Valid context_type parameters:
    - "kpi": For KPI definitions with complete business context
    - "table": For table descriptions with ALL field information
    - "query": For similar past questions with their SQL queries
    """
    args_schema: type[BaseModel] = KnowledgeBaseSearchInput
    _knowledge_base: KnowledgeBase = PrivateAttr()

    def __init__(self, knowledge_base: KnowledgeBase, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._knowledge_base = knowledge_base
        self.tags: list[str] = [TraceType.CONTEXT]

    def _run(self, query: str, context_type: Literal["kpi", "table", "query"] = "kpi", **kwargs: Any) -> str:
        doc_type = DocumentType(context_type)
        context = self._knowledge_base.retrieve_context(query=query, doc_type=doc_type)
        return context.format()

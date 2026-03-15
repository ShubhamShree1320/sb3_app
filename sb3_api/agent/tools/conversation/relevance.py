from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class RelevanceTypeInput(BaseModel):
    query: str = Field(description="The user new query")
    summary: str = Field(description="The summary from the previous conversation")


class ConversationRelevanceTool(BaseTool):
    """Relevance of the conversation against the new query."""

    name: str = "conversation_relevance"
    description: str = (
        "Use this tool to evaluate the relevance of the prior conversation messages summarization "
        "in the context of a new user query."
    )
    args_schema: type[BaseModel] = RelevanceTypeInput
    llm: Runnable

    def _run(
        self,
        query: str,
        summary: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict:
        response = self.llm.invoke({"summary": summary, "query": query})
        return {"relevance": response}

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.messages import AnyMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SummaryTypeInput(BaseModel):
    messages: list[AnyMessage] = Field(description="The list of messages from the conversation history")


class ConversationSummaryTool(BaseTool):
    """Summarizes the conversation."""

    name: str = "conversation_summary"
    description: str = "Use this tool to generate a summary of the given list of messages from the prior conversation."
    args_schema: type[BaseModel] = SummaryTypeInput
    llm: Runnable

    def _run(
        self,
        messages: list[AnyMessage],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict:
        conversation_history = "\n".join(
            f"user: {message.content}" if message.type == "human" else f"response: {message.content}"
            for message in messages[:-1]
        )
        response = self.llm.invoke({"messages": conversation_history, "query": messages[-1].content})
        return {"summary": response.content}

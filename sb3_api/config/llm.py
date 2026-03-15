from pydantic import BaseModel, Field

from sb3_api.enums.model import BedrockModel


class BaseLLMConfig(BaseModel):
    model: BedrockModel
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    max_tokens: int = Field(default=8192, ge=0)


class AgentLLMConfig(BaseLLMConfig):
    model: BedrockModel = BedrockModel.CLAUDE_4_5_SONNET


class ContextAgentLLMConfig(BaseLLMConfig):
    model: BedrockModel = BedrockModel.CLAUDE_4_5_HAIKU


class AgentFallbackLLMConfig(BaseLLMConfig):
    model: BedrockModel = BedrockModel.CLAUDE_4_5_HAIKU


class ToolLLMConfig(BaseLLMConfig):
    model: BedrockModel = BedrockModel.CLAUDE_4_SONNET

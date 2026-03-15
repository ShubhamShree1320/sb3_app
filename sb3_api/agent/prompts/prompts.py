import logging
from enum import Enum
from pathlib import Path

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class PromptId(Enum):
    AGENT_SYSTEM_PROMPT = "agent.system"
    CONTEXT_AGENT_PROMPT = "agent.context.general"
    CONTEXT_AGENT_OUTPUT = "agent.context.output"
    RECURSION_WARNING = "agent.recursion_warning"
    RECURSION_FALLBACK = "agent.recursion_fallback"
    VISUALIZATION_DETERMINE_PLOT_TYPE = "tools.visualization.determine_plot_type"
    VISUALIZATION_GENERATE_PLOT_INSTRUCT = "tools.visualization.generate_plot_instruct"
    VISUALIZATION_GENERATE_PLOT = "tools.visualization.generate_plot"
    INFO_SQL_TOOL = "tools.info_sql"
    QUERY_SQL_TOOL = "tools.query_sql"
    QUERY_SQL_CHECKER_TOOL = "tools.query_sql_checker"
    ADAPT_QUERY = "tools.adapt_query"
    CONVERSATION_HISTORY_RELEVANCE = "tools.conversation_history.relevance"
    CONVERSATION_HISTORY_SUMMARIZATION = "tools.conversation_history.summarization"
    PARTIAL_RESULTS = "tools.partial_results"
    BUSINESS_PERSONA_CONTEXT_SUMMARY = "persona.business.context.summary"
    BUSINESS_PERSONA_CONTEXT_REASONING = "persona.business.context.reasoning"
    BUSINESS_PERSONA_SYSTEM_INSTRUCTIONS = "persona.business.system"

    ANALYST_PERSONA_CONTEXT_SUMMARY = "persona.analyst.context.summary"
    ANALYST_PERSONA_CONTEXT_REASONING = "persona.analyst.context.reasoning"
    ANALYST_PERSONA_SYSTEM_INSTRUCTIONS = "persona.analyst.system"


class PromptRegistry:
    def __init__(self, prompts_yaml: str = "prompts.yaml") -> None:
        self.file = Path(__file__).parent / prompts_yaml

        with self.file.open() as fp:
            content = fp.read()
            self._prompts = yaml.safe_load(content)

    def get_prompt(self, prompt_id: PromptId) -> str:
        keys = prompt_id.value.split(".")
        prompt = self._prompts

        if prompt:
            for key in keys:
                try:
                    prompt = prompt[key]
                except KeyError as e:
                    msg = f"Missing key in {self.file.name} for {prompt_id.name}: {'.'.join(keys)}"
                    raise KeyError(msg) from e

        return prompt

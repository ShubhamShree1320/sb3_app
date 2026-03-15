from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool

from sb3_api.agent.llm import create_llm
from sb3_api.agent.prompts.prompts import PromptId, PromptRegistry
from sb3_api.agent.tools.conversation.relevance import ConversationRelevanceTool
from sb3_api.agent.tools.conversation.summarization import ConversationSummaryTool
from sb3_api.agent.tools.knowledge_base.knowledge_base import KnowledgeBase
from sb3_api.agent.tools.knowledge_base.search_knowledge_base import KnowledgeBaseSearchTool
from sb3_api.agent.tools.partial_results import PartialResultsTool
from sb3_api.agent.tools.sql import (
    InfoRedshiftSQLDatabaseTool,
    ListRedshiftSQLDatabaseTool,
    QueryRedshiftSQLCheckerTool,
    QueryRedshiftSQLDatabaseTool,
)
from sb3_api.agent.tools.sql_adaptor import AdaptQueryTool
from sb3_api.agent.tools.visualization.determine_plot_type import DeterminePlotTypeTool
from sb3_api.agent.tools.visualization.generate_plot_data import GeneratePlotTool
from sb3_api.config.llm import ToolLLMConfig
from sb3_api.repository.agent.base import BaseSQLDatabase
from sb3_api.settings import ServiceSettings


class ToolFactory:
    """Factory for creating database tools with prompts from PromptRegistry."""

    def __init__(
        self,
        db: BaseSQLDatabase,
        prompt_registry: PromptRegistry,
        knowledge_base: KnowledgeBase,
        settings: ServiceSettings,
    ) -> None:
        self.db = db
        self.llm = create_llm(config=ToolLLMConfig())
        self.prompt_registry = prompt_registry
        self.kb = knowledge_base
        self.settings = settings

    def get_sql_tools(self) -> list[BaseTool]:
        """Create SQL database tools with descriptions from PromptRegistry."""
        list_sql_tool = self.get_list_tool()
        info_sql_tool = self.get_info_tool(list_sql_tool)
        query_sql_tool = self.get_query_tool(info_sql_tool)
        query_checker_tool = self.get_checker_tool(query_sql_tool)

        return [
            query_sql_tool,
            info_sql_tool,
            list_sql_tool,
            query_checker_tool,
        ]

    def get_plot_tools(self) -> list[BaseTool]:
        plot_type_tool = self.get_determine_plot_type_tool()
        generate_plot_tool = self.get_generate_plot_tool(plot_type_tool)
        return [plot_type_tool, generate_plot_tool]

    def get_list_tool(self) -> ListRedshiftSQLDatabaseTool:
        return ListRedshiftSQLDatabaseTool(db=self.db)

    def get_info_tool(self, list_tool: ListRedshiftSQLDatabaseTool) -> InfoRedshiftSQLDatabaseTool:
        description = self.prompt_registry.get_prompt(PromptId.INFO_SQL_TOOL).format(list_tool_name=list_tool.name)
        return InfoRedshiftSQLDatabaseTool(db=self.db, description=description)

    def get_query_tool(self, info_tool: InfoRedshiftSQLDatabaseTool) -> QueryRedshiftSQLDatabaseTool:
        description = self.prompt_registry.get_prompt(PromptId.QUERY_SQL_TOOL).format(info_tool_name=info_tool.name)
        return QueryRedshiftSQLDatabaseTool(db=self.db, description=description)

    def get_checker_tool(self, query_tool: QueryRedshiftSQLDatabaseTool) -> QueryRedshiftSQLCheckerTool:
        description = self.prompt_registry.get_prompt(PromptId.QUERY_SQL_CHECKER_TOOL).format(
            query_tool_name=query_tool.name
        )
        return QueryRedshiftSQLCheckerTool(db=self.db, llm=self.llm, description=description)

    def get_determine_plot_type_tool(self) -> DeterminePlotTypeTool:
        prompt_template = ChatPromptTemplate.from_template(
            self.prompt_registry.get_prompt(PromptId.VISUALIZATION_DETERMINE_PLOT_TYPE)
        )
        llm_chain = prompt_template | self.llm
        return DeterminePlotTypeTool(llm=llm_chain)

    def get_partial_results_tool(self) -> PartialResultsTool:
        prompt_template = ChatPromptTemplate.from_template(self.prompt_registry.get_prompt(PromptId.PARTIAL_RESULTS))
        llm_chain = prompt_template | self.llm
        return PartialResultsTool(llm=llm_chain)

    def get_generate_plot_tool(self, determine_plot_tool: DeterminePlotTypeTool) -> GeneratePlotTool:
        description = self.prompt_registry.get_prompt(PromptId.VISUALIZATION_GENERATE_PLOT_INSTRUCT).format(
            determine_plot_tool_name=determine_plot_tool.name
        )
        prompt_template = ChatPromptTemplate.from_template(
            self.prompt_registry.get_prompt(PromptId.VISUALIZATION_GENERATE_PLOT)
        )
        llm_chain = prompt_template | self.llm
        return GeneratePlotTool(llm=llm_chain, description=description)

    def get_conversation_summary_tool(self) -> ConversationSummaryTool:
        prompt_template = ChatPromptTemplate.from_template(
            self.prompt_registry.get_prompt(PromptId.CONVERSATION_HISTORY_SUMMARIZATION)
        )
        llm_chain = prompt_template | self.llm
        return ConversationSummaryTool(llm=llm_chain)

    def get_conversation_relevance_tool(self) -> ConversationRelevanceTool:
        prompt_template = ChatPromptTemplate.from_template(
            self.prompt_registry.get_prompt(PromptId.CONVERSATION_HISTORY_RELEVANCE)
        )
        llm_chain = prompt_template | self.llm
        return ConversationRelevanceTool(llm=llm_chain)

    def get_knowledge_base_tool(self) -> KnowledgeBaseSearchTool:
        return KnowledgeBaseSearchTool(knowledge_base=self.kb)

    def get_adapt_query_tool(self) -> AdaptQueryTool:
        prompt_template = ChatPromptTemplate.from_template(self.prompt_registry.get_prompt(PromptId.ADAPT_QUERY))
        llm_chain = prompt_template | self.llm
        return AdaptQueryTool(llm=llm_chain)

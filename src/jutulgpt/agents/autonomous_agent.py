from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode

from jutulgpt.agents.agent_base import BaseAgent
from jutulgpt.configuration import BaseConfiguration, cli_mode
from jutulgpt.tools import (
    execute_terminal_command,
    get_working_directory,
    grep_search,
    list_files_in_directory,
    read_from_file,
    retrieve_function_documentation,
    retrieve_jutuldarcy_examples,
    run_julia_code,
    run_julia_linter,
    write_to_file,
)


class AutonomousAgent(BaseAgent):
    def __init__(
        self,
        tools: Optional[
            Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode]
        ] = None,
        print_chat_output: bool = True,
    ):
        # Set default empty tools if none provided
        if tools is None:
            tools = []

        # Initialize the base agent
        super().__init__(
            tools=tools,
            name="AutonomousAgent",
            printed_name="Agent",
            print_chat_output=print_chat_output,
        )

        self.user_provided_feedback = False

    def build_graph(self):
        """Build the react agent graph."""

        workflow = self._initialize_workflow()

        # Add nodes
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)

        workflow = self._configure_entry_point(workflow)

        # Add edges
        workflow.add_edge("tools", "agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {"tools": "tools", "continue": "get_user_input" if cli_mode else END},
        )

        # Compile with memory if standalone
        return workflow.compile()

    def get_model_from_config(
        self, config: RunnableConfig
    ) -> Union[str, LanguageModelLike]:
        configuration = BaseConfiguration.from_runnable_config(config)
        return configuration.autonomous_agent_model

    def get_prompt_from_config(self, config: RunnableConfig) -> str:
        """
        Get the prompt from the configuration.

        Returns:
            A string containing the spesific prompt from the config
        """
        configuration = BaseConfiguration.from_runnable_config(config)
        return configuration.autonomous_agent_prompt


autonomous_agent = AutonomousAgent(
    tools=[
        execute_terminal_command,
        run_julia_code,
        run_julia_linter,
        get_working_directory,
        list_files_in_directory,
        read_from_file,
        write_to_file,
        grep_search,
        retrieve_function_documentation,
        retrieve_jutuldarcy_examples,
    ],
    print_chat_output=True,
)
autonomous_agent_graph = autonomous_agent.graph

if __name__ == "__main__":
    autonomous_agent.run()

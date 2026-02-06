from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Literal, Optional, Sequence, Union, cast

from langchain_core.language_models import BaseChatModel, LanguageModelLike
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import (
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableSequence,
)
from langchain_core.tools import BaseTool
from langgraph.errors import ErrorCode, create_error_message
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.utils.runnable import RunnableCallable

import jutulgpt.state as state
from jutulgpt.cli import colorscheme, show_startup_screen, stream_to_console
from jutulgpt.configuration import LLM_TEMPERATURE, RECURSION_LIMIT
from jutulgpt.globals import console
from jutulgpt.state import State
from jutulgpt.utils import get_provider_and_model


class BaseAgent(ABC):
    """
    Abstract base class for all agent types.

    Provides common functionality like model setup, tool processing,
    prompt handling, and utility methods. Child classes must implement
    their own build_graph method to define the specific workflow.
    """

    def __init__(
        self,
        tools: Union[Sequence[Union[BaseTool, Callable, dict[str, Any]]], ToolNode],
        name: Optional[str] = None,
        printed_name: Optional[str] = "",
        part_of_multi_agent: Optional[bool] = False,
        print_chat_output: bool = True,
    ):
        if name is not None and (" " in name or not name):
            raise ValueError("Agent name must not be empty or contain spaces.")

        self.part_of_multi_agent = part_of_multi_agent
        self.name = name or self.__class__.__name__
        self.printed_name = printed_name if printed_name else name
        self.state_schema = state.State
        self.print_chat_output = print_chat_output

        # Process tools
        if isinstance(tools, ToolNode):
            self.tool_classes = list(tools.tools_by_name.values())
            self.tool_node = tools
        else:
            # Filter out built-in tools (dicts) and create ToolNode with the rest
            self.tool_node = ToolNode([t for t in tools if not isinstance(t, dict)])
            self.tool_classes = list(self.tool_node.tools_by_name.values())

        # Check which tools return direct
        self.should_return_direct = {
            t.name for t in self.tool_classes if t.return_direct
        }

        # Build and compile the graph (implemented by child classes)
        self.graph = self.build_graph()

        # WARNING: This requires connection to internet. Therefore it is currently commented out.
        # self.generate_graph_visualization()

    @abstractmethod
    def get_prompt_from_config(self, config: RunnableConfig) -> str:
        """
        Get the prompt from the configuration.

        Returns:
            A string containing the spesific prompt from the config
        """
        pass

    @abstractmethod
    def get_model_from_config(
        self, config: RunnableConfig
    ) -> Union[str, LanguageModelLike]:
        """
        Get the model-name from the configuration.

        Returns:
            A string containing the spesific prompt from the config
        """
        pass

    @abstractmethod
    def build_graph(self) -> Any:
        """
        Build the graph for the agent.

        Returns:
            A compiled StateGraph instance representing the agent's workflow.


        Example:
        Building a ReAct agent (https://langchain-ai.github.io/langgraph/how-tos/react-agent-from-scratch/#define-nodes-and-edges)
        ```
        workflow = StateGraph(state.State, config_schema=BaseConfiguration)

        # Define the two nodes we will cycle between
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)

        if not self.part_of_multi_agent and cli_mode:
            workflow.add_node("get_user_input", self.get_user_input)
            workflow.set_entry_point("get_user_input")
            workflow.add_edge("get_user_input", "agent")
        else:
            workflow.set_entry_point("agent")

        # We now add a conditional edge
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "continue": "get_user_input"
                if not self.part_of_multi_agent and cli_mode
                else END,
            },
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile(name=self.name)
        ```
        """
        pass

    def _get_chat_model(self, model: Union[str, LanguageModelLike]) -> BaseChatModel:
        """Setup and bind tools to the model."""
        if isinstance(model, str):
            try:
                from langchain.chat_models import init_chat_model
            except ImportError:
                raise ImportError("Please install langchain to use string model names")

            provider, model_name = get_provider_and_model(model)

            if (
                provider == "ollama" and model_name == "qwen3:14b"
            ):  # WARNING: This is bad practice!
                chat_model = init_chat_model(
                    model_name,
                    model_provider=provider,
                    temperature=LLM_TEMPERATURE,
                    reasoning=True,
                    streaming=True,
                )
            else:
                chat_model = init_chat_model(
                    model_name,
                    model_provider=provider,
                    temperature=LLM_TEMPERATURE,
                    streaming=True,
                )
            model = cast(BaseChatModel, chat_model)

        # Get the underlying model
        if isinstance(model, RunnableSequence):
            model = next(
                (
                    step
                    for step in model.steps
                    if isinstance(step, (RunnableBinding, BaseChatModel))
                ),
                model,
            )

        if isinstance(model, RunnableBinding):
            model = model.bound

        if not isinstance(model, BaseChatModel):
            raise TypeError(f"Expected model to be a ChatModel, got {type(model)}")

        return cast(BaseChatModel, model)

    def _load_model(self, config: RunnableConfig) -> BaseChatModel:
        """Load the model from the name specified in the configuration."""
        chat_model = self._get_chat_model(self.get_model_from_config(config=config))
        if self._should_bind_tools(chat_model):
            chat_model = chat_model.bind_tools(self.tool_classes)
        return cast(BaseChatModel, chat_model)

    def generate_graph_visualization(self):
        """Generate mermaid visualization of the graph."""
        try:
            filename = f"./{self.name.lower()}_graph.png"
            self.graph.get_graph().draw_mermaid_png(output_file_path=filename)
        except Exception as e:
            # Don't fail if visualization generation fails
            print(f"Warning: Could not generate graph visualization: {e}")

    def invoke_model(
        self,
        state: state.State,
        config: RunnableConfig,
        messages_list: Optional[List] = None,
    ) -> AIMessage:
        """Invoke the model with the given prompt and state."""
        model = self._load_model(config=config)

        from jutulgpt.rag.package_path import get_package_root

        jd_root = get_package_root()
        jd_path = str(jd_root) if jd_root else "NOT INSTALLED"
        workspace_message = f"**Current workspace:** {os.getcwd()} \n**JutulDarcy documentation and examples can be found at:** {jd_path}"

        if not messages_list:
            messages_list: List = [
                SystemMessage(content=self.get_prompt_from_config(config=config)),
                SystemMessage(content=workspace_message),
            ]
            trimmed_state_messages = self._trim_state_messages(state.messages, model)
            messages_list.extend(trimmed_state_messages)

        # Invoke the model
        if self.print_chat_output:
            chat_response = stream_to_console(
                llm=model,
                message_list=messages_list,
                config=config,
                title=self.printed_name,
                border_style=colorscheme.normal,
            )

            response = cast(AIMessage, chat_response)
        else:
            response = cast(AIMessage, model.invoke(messages_list, config))

        # Add agent name to the response
        response.name = self.name

        return response

    def _should_bind_tools(self, model: BaseChatModel) -> bool:
        """Check if we need to bind tools to the model."""
        if len(self.tool_classes) == 0:
            return False

        if isinstance(model, RunnableBinding):
            if "tools" in model.kwargs:
                bound_tools = model.kwargs["tools"]
                if len(self.tool_classes) != len(bound_tools):
                    raise ValueError(
                        f"Number of tools mismatch. Expected {len(self.tool_classes)}, got {len(bound_tools)}"
                    )
                return False
        return True

    def _get_prompt_runnable(
        self, prompt: Optional[Union[SystemMessage, str]]
    ) -> Runnable:
        """
        Create a prompt runnable from the prompt.

        Note: This is currently not used, but we should movefrom the get_prompt_from_config function to this method
        """
        if prompt is None:
            return RunnableCallable(lambda state: state.messages, name="Prompt")
        elif isinstance(prompt, str):
            system_message = SystemMessage(content=prompt)
            return RunnableCallable(
                lambda state: [system_message] + list(state.messages), name="Prompt"
            )
        elif isinstance(prompt, SystemMessage):
            return RunnableCallable(
                lambda state: [prompt] + list(state.messages), name="Prompt"
            )
        else:
            raise ValueError(f"Got unexpected type for prompt: {type(prompt)}")

    def _validate_chat_history(self, messages: Sequence[BaseMessage]) -> None:
        """Validate that all tool calls have corresponding tool messages."""
        all_tool_calls = [
            tool_call
            for message in messages
            if isinstance(message, AIMessage)
            for tool_call in message.tool_calls
        ]
        tool_call_ids_with_results = {
            message.tool_call_id
            for message in messages
            if isinstance(message, ToolMessage)
        }
        tool_calls_without_results = [
            tool_call
            for tool_call in all_tool_calls
            if tool_call["id"] not in tool_call_ids_with_results
        ]
        if tool_calls_without_results:
            error_message = create_error_message(
                message="Found AIMessages with tool_calls that do not have corresponding ToolMessage.",
                error_code=ErrorCode.INVALID_CHAT_HISTORY,
            )
            raise ValueError(error_message)

    def _are_more_steps_needed(self, state: state.State, response: BaseMessage) -> bool:
        """Check if more steps are needed based on remaining steps and tool calls."""
        has_tool_calls = isinstance(response, AIMessage) and bool(response.tool_calls)
        all_tools_return_direct = (
            all(
                call["name"] in self.should_return_direct
                for call in response.tool_calls
            )
            if isinstance(response, AIMessage) and response.tool_calls
            else False
        )
        remaining_steps = state.remaining_steps
        is_last_step = state.is_last_step

        return (
            (remaining_steps is None and is_last_step and has_tool_calls)
            or (
                remaining_steps is not None
                and remaining_steps < 1
                and all_tools_return_direct
            )
            or (remaining_steps is not None and remaining_steps < 2 and has_tool_calls)
        )

    def _trim_state_messages(
        self,
        messages: Sequence[BaseMessage],
        model: Union[BaseChatModel, Runnable[LanguageModelInput, BaseMessage]],
    ) -> Sequence[BaseMessage]:
        trimmed_state_messages = trim_messages(
            messages,
            max_tokens=40000,  # adjust for model's context window minus system & files message
            strategy="last",
            token_counter=model,
            include_system=False,  # Not needed since systemMessage is added separately
            allow_partial=True,
        )
        return trimmed_state_messages

    def should_continue(self, state: state.State) -> Literal["tools", "continue"]:
        """
        Commonly used function for conditional edges. Checks is the model has used tools or not.
        """
        messages = state.messages
        last_message = messages[-1]

        # If the last message has tool calls, go to tools
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        else:
            return "continue"

    def call_model(self, state: state.State, config: RunnableConfig) -> dict:
        """Call the model with the current state."""
        response = self.invoke_model(state=state, config=config)
        return {"messages": [response]}

    def get_user_input(self, state: state.State, config: RunnableConfig) -> dict:
        """Get user input for standalone mode."""

        user_input = ""
        while not user_input:  # Handle empty input
            console.print("[bold blue]User Input:[/bold blue] ")
            user_input = console.input("> ")

        # Check for quit command
        if user_input.strip().lower() in ["q", "quit"]:
            console.print("[bold red]Goodbye![/bold red]")
            exit(0)

        return {
            "messages": [HumanMessage(content=user_input)],
        }

    def state_from_mcp_input(self, state: State, config: RunnableConfig) -> dict:
        """
        Convert from the input from Copilot to a question that JutulGPT can interpret. Used when running an MCP-server for VSCode/Copilot integration.
        """

        question = state.mcp_question

        try:
            current_filepath = state.mcp_current_filepath
        except:
            current_filepath = ""

        if not current_filepath:
            current_filepath = "Filepath not provided"

        full_question = f"""
You are called as a tool by another agent. Try to answer the question, and note that the other agent can only read your final ouput.

The current file we are working in. You should read its content before trying to respond: {current_filepath}

Here is the question asked by the other agent:
{question}
"""
        return {"messages": [full_question]}

    def run(self) -> None:
        """Run the agent."""
        from jutulgpt.configuration import init

        init()

        if self.part_of_multi_agent:
            raise ValueError("Cannot run standalone mode when part_of_multi_agent=True")

        try:
            show_startup_screen()

            # Create configuration
            config = RunnableConfig(configurable={}, recursion_limit=RECURSION_LIMIT)

            # Create initial state conforming to the state schema
            initial_state = {
                "messages": [],  # Start with empty messages, user input will add the first message
                "remaining_steps": RECURSION_LIMIT,
                "is_last_step": False,
            }

            # The graph will handle the looping internally
            self.graph.invoke(initial_state, config=config)

        except KeyboardInterrupt:
            console.print("\n[bold red]Goodbye![/bold red]")

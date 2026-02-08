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
    RemoveMessage,
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
from jutulgpt.cli import (
    colorscheme,
    show_startup_screen,
    stream_to_console,
)
from jutulgpt.configuration import (
    CONTEXT_TRIM_THRESHOLD,
    RECENT_MESSAGES_TO_KEEP,
    RECURSION_LIMIT,
    BaseConfiguration,
    cli_mode,
)
from jutulgpt.context import ContextTracker, summarize_conversation
from jutulgpt.globals import console
from jutulgpt.logging import SessionLogger, set_session_logger
from jutulgpt.state import State
from jutulgpt.utils.code_parsing import get_code_from_response
from jutulgpt.utils.model import (
    get_message_text,
    load_chat_model,
)


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
        self._logger: Optional[SessionLogger] = None
        self._context_tracker: Optional[ContextTracker] = None
        self._conversation_summary: str = (
            ""  # Summary of the conversation if summarization has happened
        )
        self._messages_to_remove: List[
            RemoveMessage
        ] = []  # Messages to delete from state

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

    @staticmethod
    def _ensure_string_content(msg: BaseMessage) -> BaseMessage:
        """Return a message with `content` converted to a plain string.

        Some providers (notably OpenAI Responses) may return structured blocks in
        `message.content`. This breaks downstream token counting / trimming which
        expects a string. We store only the plain-text view in state/history.
        """
        if not hasattr(msg, "content") or isinstance(getattr(msg, "content"), str):
            return msg

        text = get_message_text(msg)

        # Pydantic v2 (LangChain >=0.3): immutable copy
        try:
            return msg.model_copy(update={"content": text})  # type: ignore[attr-defined]
        except Exception:
            # Fallback: in-place assignment (best effort)
            try:
                msg.content = text  # type: ignore[attr-defined]
            except Exception:
                pass
            return msg

    def _apply_cli_model_selection(self) -> bool:
        """Apply `--model` preset selection for CLI runs.

        Returns False if argparse handled `-h/--help` (i.e. should exit).
        """
        if not cli_mode:
            return True

        from jutulgpt.cli.model_cli import apply_model_from_cli, parse_cli_args

        try:
            args = parse_cli_args()
        except SystemExit:
            # argparse handled -h/--help
            return False

        apply_model_from_cli(args.model)
        return True

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
            model = cast(BaseChatModel, load_chat_model(model))

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
        # Fallback logger init for non-CLI modes (e.g., MCP)
        if self._logger is None:
            configuration = BaseConfiguration.from_runnable_config(config=config)
            self._logger = SessionLogger.from_config(configuration)
            set_session_logger(self._logger)

        model = self._load_model(config=config)
        configuration = BaseConfiguration.from_runnable_config(config=config)

        system_prompt = self.get_prompt_from_config(config=config)
        from jutulgpt.rag.package_paths import get_package_root

        try:
            jutuldarcy_path = str(get_package_root("JutulDarcy"))
        except Exception:
            jutuldarcy_path = "(unable to resolve – is JutulDarcy installed?)"
        workspace_message = f"**Current workspace:** {os.getcwd()} \n**JutulDarcy documentation and examples are read from the installed package at:** {jutuldarcy_path}"

        # Initialize context tracker
        if self._context_tracker is None:
            self._context_tracker = ContextTracker(
                system_prompt=system_prompt + workspace_message,
                tool_definitions=self.tool_classes,
                model=model,
                max_tokens=configuration.context_window_size,
            )

        if not messages_list:
            # Prepare context (builds messages, summarizes if needed)
            messages_list = self._prepare_context(
                system_prompt, workspace_message, state.messages, model, config
            )

        # Invoke the model
        if self.print_chat_output:
            chat_response = stream_to_console(
                llm=model,
                message_list=messages_list,
                config=config,
                title=self.printed_name,
                border_style=colorscheme.normal,
                logger=self._logger,
            )

            response = cast(AIMessage, chat_response)
        else:
            response = cast(AIMessage, model.invoke(messages_list, config))
            # Log non-streamed responses
            if self._logger and self._logger.enabled:
                # response.content can be str or list, convert to str for logging
                content = (
                    response.content
                    if isinstance(response.content, str)
                    else str(response.content)
                )
                self._logger.log_assistant(
                    content=content if content else "",
                    title=self.printed_name or "Assistant",
                    tool_calls=getattr(response, "tool_calls", None),
                )

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

    def _prepare_context(
        self,
        system_prompt: str,
        workspace_message: str,
        state_messages: Sequence[BaseMessage],
        model: Union[BaseChatModel, Runnable[LanguageModelInput, BaseMessage]],
        config: RunnableConfig,
    ) -> List[BaseMessage]:
        """Prepare context for model invocation.

        Builds message list, compresses old messages if context is too high,
        and returns the final message list for the model.
        """
        configuration = BaseConfiguration.from_runnable_config(config=config)
        all_messages = [self._ensure_string_content(m) for m in state_messages]
        summarization_happened = False

        # Compress old messages if context usage exceeds threshold
        if self._context_tracker:
            usage = self._context_tracker.update(all_messages)

            if usage.usage_fraction >= configuration.context_summarize_threshold:
                cutoff = self._find_compression_cutoff(all_messages)

                if cutoff > 0:
                    summary = summarize_conversation(
                        all_messages[:cutoff],
                        model,
                        config,
                        previous_summary=self._conversation_summary,
                    )
                    if summary:
                        self._conversation_summary = summary
                        summarization_happened = True

                        # Mark old messages for removal from LangGraph state
                        self._messages_to_remove = [
                            RemoveMessage(id=msg.id)
                            for msg in all_messages[:cutoff]
                            if msg.id is not None
                        ]

                        # Log compression
                        log_msg = f"⚡ Context at {usage.usage_percent:.0f}% - compressed {cutoff} messages."
                        console.print(f"[yellow]{log_msg}[/yellow]")
                        if self._logger and self._logger.enabled:
                            self._logger._write_raw(f"**{log_msg}**\n\n---\n\n")

                        all_messages = all_messages[cutoff:]

        # Combine all system content into one SystemMessage because
        # trim_messages with include_system=True only preserves the first SystemMessage
        system_parts = [system_prompt, workspace_message]

        if self._conversation_summary:
            system_parts.append(
                f"## Previous conversation summary:\n{self._conversation_summary}"
            )

        if summarization_happened:
            system_parts.append(
                "[Context was just summarized. The following messages contain your latest work. Continue with your current task.]"
            )

        combined_system_content = "\n\n".join(system_parts)
        messages_list: List[BaseMessage] = [
            SystemMessage(content=combined_system_content),
        ]

        messages_list.extend(all_messages)

        # Update tracker with final list
        if self._context_tracker:
            self._context_tracker.update(
                messages_list, summary=self._conversation_summary
            )

        # Trim as safety net (in case summarization didn't compress enough)
        trim_limit = int(configuration.context_window_size * CONTEXT_TRIM_THRESHOLD)
        final_messages = list(
            trim_messages(
                messages_list,
                max_tokens=trim_limit,
                strategy="last",
                token_counter=model,
                include_system=True,
                start_on="human",
                end_on=("human", "tool"),
                allow_partial=False,
            )
        )

        return final_messages

    def _find_compression_cutoff(self, messages: List[BaseMessage]) -> int:
        """Find cutoff for compression, preserving recent messages and tool call boundaries."""
        cutoff = len(messages) - RECENT_MESSAGES_TO_KEEP

        # Don't compress past the last HumanMessage
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                cutoff = min(cutoff, i)
                break

        # Don't cut after AIMessage with tool_calls (would orphan the ToolMessages)
        while cutoff > 0 and isinstance(messages[cutoff - 1], AIMessage):
            if getattr(messages[cutoff - 1], "tool_calls", None):
                cutoff -= 1
            else:
                break

        return max(0, cutoff)

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

    def _finalize_context(self, response: AIMessage) -> List[BaseMessage]:
        """Finalize context after model call, bundling response with pending state changes.

        Pairs with _prepare_context. Subclasses should use this when overriding
        call_model to ensure RemoveMessage objects from compression are included.
        """
        updates: List[BaseMessage] = []
        if self._messages_to_remove:
            updates.extend(self._messages_to_remove)
            self._messages_to_remove = []

        # Store a plain-text version of the assistant message in state to avoid
        # carrying provider-specific structured blocks into later token counting.
        response = cast(AIMessage, self._ensure_string_content(response))

        updates.append(response)
        return updates

    def call_model(self, state: State, config: RunnableConfig) -> dict:
        """Call the model with the current state."""

        response = self.invoke_model(state=state, config=config)

        # Check if we need more steps
        if self._are_more_steps_needed(state, response):
            fallback = AIMessage(
                id=response.id,
                content="Sorry, need more steps to process this request.",
            )
            return {"messages": self._finalize_context(fallback)}

        # With OpenAI Responses API, response.content may be a list of content blocks.
        # Always use the normalized text view for downstream parsing.
        response_text = get_message_text(response)
        code_block = get_code_from_response(response=response_text)

        # Finalize context: bundle response with any pending state changes
        messages = self._finalize_context(response)
        return {
            "messages": messages,
            "code_block": code_block,
            "error": False,
            "mcp_answer": response_text,
        }

    def get_user_input(self, state: state.State, config: RunnableConfig) -> dict:
        """Get user input for standalone mode."""
        configuration = BaseConfiguration.from_runnable_config(config=config)

        # Update and display context usage before user input
        if self._context_tracker:
            # Re-count with current state (includes model's latest response)
            self._context_tracker.update(
                list(state.messages), summary=self._conversation_summary
            )
            self._context_tracker.display(
                logger=self._logger,
                show_console=configuration.show_context_usage,
                console_threshold=configuration.context_display_threshold,
            )

        user_input = ""
        while not user_input:  # Handle empty input
            console.print("[bold blue]User Input:[/bold blue] ")
            user_input = console.input("> ")

        # Check for quit command
        if user_input.strip().lower() in ["q", "quit"]:
            console.print("[bold red]Goodbye![/bold red]")
            exit(0)

        # Log user input
        if self._logger and self._logger.enabled:
            self._logger.log_user(content=user_input, title="User")

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
        if self.part_of_multi_agent:
            raise ValueError("Cannot run standalone mode when part_of_multi_agent=True")

        try:
            # CLI model selection (before config/logger initialization)
            if not self._apply_cli_model_selection():
                return

            show_startup_screen()

            # Create configuration
            config = RunnableConfig(configurable={}, recursion_limit=RECURSION_LIMIT)
            configuration = BaseConfiguration.from_runnable_config(config=config)

            # Initialize session logger once at startup
            self._logger = SessionLogger.from_config(configuration)
            set_session_logger(self._logger)

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

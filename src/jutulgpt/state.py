"""
State structures for the agent, including:
- InputState: initial and ongoing message state for the agent's execution.
- State: main agent state, tracks errors, iterations, and step control.
- CodeBlock: container for code and imports, with formatting helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Sequence


class CodeBlock(BaseModel):
    """
    Container for a code block, split into imports and main code.

    - imports: Any import statements needed for the code to run.
    - code: The main code body (excluding imports).

    """

    imports: str = Field(default="", description="Code block import statements")
    code: str = Field(
        default="", description="Code block not including import statements"
    )

    def is_empty(self) -> bool:
        if not self.imports and not self.code:
            return True
        return False

    def get_full_code(
        self, within_julia_context: bool = False, return_empty_if_no_code: bool = False
    ) -> str:
        """
        Returns the full code block, optionally wrapped as a Julia markdown code block.
        If within_julia_context is True, wraps the code in triple backticks and 'julia' for syntax highlighting.
        """

        # WARNING: This might be unneccessary
        if return_empty_if_no_code and not self.imports and not self.code:
            return ""

        full_code = "```julia" if within_julia_context else ""
        if self.imports:
            full_code += "\n" if within_julia_context else ""
            full_code += f"{self.imports}"
        if self.code:
            full_code += "\n" if within_julia_context or self.imports else ""
            full_code += f"{self.code}"
        full_code += "\n```" if within_julia_context else ""
        return full_code


# Define minimal input schema for MCP
@dataclass
class MCPInputState:
    """
    Input schema for MCP endpoint, containing only the user question or prompt.
    """

    question: str
    current_filepath: str


# Define minimal output schema for MCP
@dataclass
class MCPOutputState:
    """
    Output schema for MCP endpoint, containing the agent's response.
    """

    answer: str
    code_block: CodeBlock  # Include if code output is part of the MCP response


# Define internal state for graph processing
@dataclass
class State:
    """
    Internal state for managing the agent's conversation and tool interactions.
    Extends the input/output schemas with additional fields for internal use.
    """

    mcp_question: str = ""
    mcp_current_filepath: str = ""
    mcp_answer: str = ""
    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    error: bool = field(default=False)
    error_message: str = field(default="")
    iterations: int = field(default=0)
    regenerate_code: bool = field(default=False)
    retrieved_context: str = field(default="")
    code_block: CodeBlock = field(default_factory=CodeBlock)
    is_last_step: bool = field(default=False)
    remaining_steps: int = field(default=50)

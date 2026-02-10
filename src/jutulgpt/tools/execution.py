"""Tools for executing code and terminal commands."""

from __future__ import annotations

import os
import subprocess

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from jutulgpt.cli import colorscheme, print_to_console
from jutulgpt.configuration import OUTPUT_TRUNCATION_LIMIT
from jutulgpt.logging import ToolEntry, get_session_logger
from jutulgpt.nodes.check_code import _run_julia_code, _run_linter
from jutulgpt.utils.code_transforms import shorter_simulations


def _truncate_output(
    text: str, max_length: int = OUTPUT_TRUNCATION_LIMIT
) -> tuple[str, bool]:
    """Truncate text to max_length. Returns (truncated_text, was_truncated)."""
    if len(text) <= max_length:
        return text, False
    return text[:max_length] + "...", True


def _format_subprocess_output(
    result: subprocess.CompletedProcess, include_exit_code: bool = True
) -> str:
    """Format subprocess result with truncation. Returns formatted output string."""
    parts = []
    truncation_info = []

    if result.stdout:
        stdout_truncated, was_truncated = _truncate_output(result.stdout)
        parts.append(f"STDOUT:\n{stdout_truncated}")
        if was_truncated:
            truncation_info.append(
                f"stdout: {len(result.stdout):,} → {OUTPUT_TRUNCATION_LIMIT:,} chars"
            )

    if result.stderr:
        stderr_truncated, was_truncated = _truncate_output(result.stderr)
        parts.append(f"STDERR:\n{stderr_truncated}")
        if was_truncated:
            truncation_info.append(
                f"stderr: {len(result.stderr):,} → {OUTPUT_TRUNCATION_LIMIT:,} chars"
            )

    if include_exit_code and result.returncode != 0:
        parts.append(f"EXIT CODE: {result.returncode}")

    if truncation_info:
        parts.append(f"[Output truncated: {'; '.join(truncation_info)}]")

    return "\n\n".join(parts)


class RunJuliaCodeInput(BaseModel):
    code: str = Field(
        description="The Julia code that should be executed",
    )


@tool(
    "run_julia_code",
    args_schema=RunJuliaCodeInput,
    description="Execute Julia code. Returns output or error message.",
)
def run_julia_code(code: str):
    #code = shorter_simulations(code)
    out, code_failed = _run_julia_code(code, print_code=True)
    if code_failed:
        return out
    return "Code executed successfully!"


class RunJuliaLinterInput(BaseModel):
    code: str = Field(
        description="The Julia code that should be analyzed using the linter",
    )


@tool(
    "run_julia_linter",
    args_schema=RunJuliaLinterInput,
    description="Run a static analysis of Julia code using a linter. Returns output or error message.",
)
def run_julia_linter(code: str):
    out, issues_found = _run_linter(code)
    if issues_found:
        return out
    return "Linter found no issues."


@tool("execute_terminal_command", parse_docstring=True)
def execute_terminal_command(command: str) -> str:
    """
    Execute a terminal command and return the output. Remember to include the project directory in the command when running the julia command. I.e. write f.ex. `julia --project=. my_script.jl`

    Args:
        command: The command to execute. IMPORTANT to remember to add the project directory to the command when running Julia!

    Returns:
        str: The output from executing the command (stdout and stderr combined)
    """

    from jutulgpt.human_in_the_loop import cli

    run_command, command = cli.modify_terminal_run(command)

    if not run_command:
        return "User did not allow you to run this command."

    working_directory = os.getcwd()

    try:
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
        )

        output = _format_subprocess_output(result)

        print_to_console(
            text=output,
            title="Run finished",
            border_style=colorscheme.success
            if not result.stderr
            else colorscheme.message,
        )

        # Log to session file if logger available
        logger = get_session_logger()
        if logger:
            logger.log(
                ToolEntry(
                    content=output,
                    title="Run finished",
                    tool_name="execute_terminal_command",
                    args={"command": command, "cwd": working_directory},
                    returncode=result.returncode,
                )
            )

        return (
            output.strip()
            if output.strip()
            else "Command executed successfully with no output."
        )

    except subprocess.TimeoutExpired:
        error_msg = "ERROR: Command execution timed out after 60 seconds."
        print_to_console(
            text=error_msg,
            title="Run error",
            border_style=colorscheme.error,
        )
        logger = get_session_logger()
        if logger:
            logger.log(
                ToolEntry(
                    content=error_msg,
                    title="Run error",
                    tool_name="execute_terminal_command",
                    args={"command": command, "cwd": working_directory},
                    error="timeout",
                )
            )
        return error_msg

    except Exception as e:
        error_msg = f"ERROR: Failed to execute command: {str(e)}"
        print_to_console(
            text=error_msg,
            title="Run error",
            border_style=colorscheme.error,
        )
        logger = get_session_logger()
        if logger:
            logger.log(
                ToolEntry(
                    content=error_msg,
                    title="Run error",
                    tool_name="execute_terminal_command",
                    args={"command": command, "cwd": working_directory},
                    error=str(e),
                )
            )
        return error_msg


@tool
def get_working_directory() -> str:
    """
    Get the current working directory.

    Returns:
        str: The current working directory path
    """
    return os.getcwd()

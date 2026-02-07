"""Tools for executing code and terminal commands."""

from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from jutulgpt.cli import colorscheme, print_to_console
from jutulgpt.configuration import OUTPUT_TRUNCATION_LIMIT
from jutulgpt.logging import ToolEntry, get_session_logger
from jutulgpt.nodes.check_code import _run_julia_code, _run_linter
from jutulgpt.utils import fix_imports, shorter_simulations


def _truncate_output(text: str, max_length: int = OUTPUT_TRUNCATION_LIMIT) -> tuple[str, bool]:
    """Truncate text to max_length. Returns (truncated_text, was_truncated)."""
    if len(text) <= max_length:
        return text, False
    return text[:max_length] + "...", True


def _format_subprocess_output(result: subprocess.CompletedProcess, include_exit_code: bool = True) -> str:
    """Format subprocess result with truncation. Returns formatted output string."""
    parts = []
    truncation_info = []

    if result.stdout:
        stdout_truncated, was_truncated = _truncate_output(result.stdout)
        parts.append(f"STDOUT:\n{stdout_truncated}")
        if was_truncated:
            truncation_info.append(f"stdout: {len(result.stdout):,} → {OUTPUT_TRUNCATION_LIMIT:,} chars")

    if result.stderr:
        stderr_truncated, was_truncated = _truncate_output(result.stderr)
        parts.append(f"STDERR:\n{stderr_truncated}")
        if was_truncated:
            truncation_info.append(f"stderr: {len(result.stderr):,} → {OUTPUT_TRUNCATION_LIMIT:,} chars")

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
    code = fix_imports(code)
    code = shorter_simulations(code)
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
def list_directory_contents(directory_path: str) -> str:
    """
    List the contents of a directory.

    Args:
        directory_path: Path to the directory to list

    Returns:
        str: Directory contents listing
    """
    try:
        if not os.path.exists(directory_path):
            return f"ERROR: Directory {directory_path} does not exist."

        if not os.path.isdir(directory_path):
            return f"ERROR: {directory_path} is not a directory."

        contents = os.listdir(directory_path)
        if not contents:
            return f"Directory {directory_path} is empty."

        # Sort and format the contents
        contents.sort()
        formatted_contents = []
        for item in contents:
            item_path = os.path.join(directory_path, item)
            if os.path.isdir(item_path):
                formatted_contents.append(f"[DIR]  {item}/")
            else:
                formatted_contents.append(f"[FILE] {item}")

        return f"Contents of {directory_path}:\n" + "\n".join(formatted_contents)

    except Exception as e:
        return f"ERROR: Failed to list directory contents: {str(e)}"


@tool
def get_working_directory() -> str:
    """
    Get the current working directory.

    Returns:
        str: The current working directory path
    """
    return os.getcwd()


@tool
def change_working_directory(directory_path: str) -> str:
    """
    Change the working directory.

    Args:
        directory_path: Path to the directory to change to

    Returns:
        str: Confirmation message or error
    """
    try:
        if not os.path.exists(directory_path):
            return f"ERROR: Directory {directory_path} does not exist."

        if not os.path.isdir(directory_path):
            return f"ERROR: {directory_path} is not a directory."

        os.chdir(directory_path)
        return f"Successfully changed working directory to: {os.getcwd()}"

    except Exception as e:
        return f"ERROR: Failed to change directory: {str(e)}"


@tool
def create_julia_workspace(task_name: str, base_directory: Optional[str] = None) -> str:
    """
    Create a simple workspace with a single Julia file.

    Args:
        task_name: Name of the task
        base_directory: Optional base directory to create workspace in

    Returns:
        str: Path to the created Julia file
    """
    if base_directory is None:
        base_directory = os.getcwd()

    # Simple sanitization
    safe_task_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", task_name.lower())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    workspace_dir = Path(base_directory) / "jutulgpt_workspaces"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    julia_file = workspace_dir / f"{safe_task_name}_{timestamp}.jl"

    print_to_console(
        text=f"Creating Julia workspace for task '{task_name}' at {julia_file}",
        title="Tool: Create Julia Workspace",
        border_style=colorscheme.message,
    )

    try:
        with open(julia_file, "w") as f:
            f.write(f"# {task_name}\n")
            f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        return str(julia_file)

    except Exception as e:
        return f"ERROR: Failed to create workspace: {str(e)}"


@tool
def write_julia_code_to_file(code: str, file_path: str, append: bool = False) -> str:
    """
    Write Julia code to a file.

    Args:
        code: The Julia code to write
        file_path: Path to the Julia file
        append: Whether to append to existing file or overwrite

    Returns:
        str: Confirmation message or error
    """
    print_to_console(
        text=f"Writing Julia code to {file_path} (append={append})",
        title="Tool: Write Julia Code to File",
        border_style=colorscheme.message,
    )

    try:
        mode = "a" if append else "w"
        with open(file_path, mode) as f:
            if append:
                f.write("\n")
            f.write(code)
            if not code.endswith("\n"):
                f.write("\n")

        action = "appended to" if append else "written to"
        return f"Successfully {action} {file_path}"

    except Exception as e:
        return f"ERROR: Failed to write to file: {str(e)}"


@tool
def execute_julia_file(file_path: str) -> str:
    """
    Execute a Julia file and return the output.

    Args:
        file_path: Path to the Julia file to execute

    Returns:
        str: Execution output and exit code
    """
    print_to_console(
        text=f"Executing Julia file: {file_path}",
        title="Tool: Execute Julia File",
        border_style=colorscheme.warning,
    )
    try:
        if not os.path.exists(file_path):
            return f"ERROR: File {file_path} does not exist"

        result = subprocess.run(
            ["julia", file_path], capture_output=True, text=True, timeout=30
        )

        output = f"=== Execution of {file_path} ===\n"
        output += _format_subprocess_output(result, include_exit_code=True)

        print_to_console(
            text=output.strip(),
            title="Execution Result",
            border_style=colorscheme.success
            if result.returncode == 0
            else colorscheme.error,
        )

        return output

    except subprocess.TimeoutExpired:
        return f"ERROR: Execution of {file_path} timed out after 30 seconds"
    except Exception as e:
        return f"ERROR: Failed to execute {file_path}: {str(e)}"

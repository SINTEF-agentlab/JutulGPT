from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from jutulgpt.cli import colorscheme, print_to_console
from jutulgpt.configuration import (
    DISPLAY_CONTENT_MAX_LENGTH,
    OUTPUT_TRUNCATION_LIMIT,
    BaseConfiguration,
    cli_mode,
)
from jutulgpt.julia import get_error_message, get_linting_result, run_code
from jutulgpt.logging import CodeRunnerEntry, ToolEntry, get_session_logger
from jutulgpt.state import State
from jutulgpt.utils.code_parsing import add_julia_context, get_code_from_response
from jutulgpt.utils.code_transforms import shorter_simulations


def _truncate(text: str, max_length: int = DISPLAY_CONTENT_MAX_LENGTH) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def _run_linter(code: str, print_code: bool = True) -> tuple[str, bool]:
    """
    Returns:
        str: String containing the linting issues found in the code. Empty if no issues found.
        bool: True if issues were found, False otherwise.
    """
    if print_code:
        display_code = _truncate(f"```julia\n{code}\n```")
        print_to_console(
            text="Running static analysis:\n" + display_code,
            title="Linter",
            border_style=colorscheme.warning,
        )
    else:
        print_to_console(
            text="Running static analysis...",
            title="Linter",
            border_style=colorscheme.warning,
        )

    linting_result = get_linting_result(code)

    logger = get_session_logger()
    if logger:
        if linting_result:
            logger.log(
                ToolEntry(
                    content=f"Linter found issues:\n{_truncate(linting_result)}",
                    title="Linter",
                    tool_name="run_julia_linter",
                )
            )
        else:
            logger.log(
                ToolEntry(
                    content="Linter found no issues.",
                    title="Linter",
                    tool_name="run_julia_linter",
                )
            )

    if linting_result:
        linting_message = (
            "## Linter issues found:\n"
            + "Linter returned the following issues:\n"
            + linting_result
        )
        return linting_message, True
    return "", False


def _run_julia_code(code: str, print_code: bool = True) -> tuple[str, bool]:
    """
    Returns:
        str: String containing the code running failed. Empty if the code executed successfully.
        bool: True if issues were found, False otherwise.
    """
    logger = get_session_logger()

    if print_code:
        display_code = _truncate(f"```julia\n{code}\n```")
        print_to_console(
            text="Running code:\n" + display_code,
            title="Code Runner",
            border_style=colorscheme.warning,
        )
    else:
        print_to_console(
            text="Running code...",
            title="Code Runner",
            border_style=colorscheme.warning,
        )

    # result = run_string(code)
    result = run_code(code)

    if result.get("error", False):
        julia_error_message = get_error_message(result)

        # Truncate for display/logging if message is very long
        is_truncated = len(julia_error_message) > OUTPUT_TRUNCATION_LIMIT
        display_error = _truncate(julia_error_message, OUTPUT_TRUNCATION_LIMIT)

        truncation_note = ""
        if is_truncated:
            truncation_note = f"\n\n[Truncated: {len(julia_error_message):,} chars → {OUTPUT_TRUNCATION_LIMIT:,} chars]"

        print_to_console(
            text=f"Code failed!\n\n{display_error}{truncation_note}",
            title="Code Runner",
            border_style=colorscheme.error,
        )

        # Log failure with truncated error
        if logger:
            logger.log(
                CodeRunnerEntry(
                    content=f"Code failed!\n\n{display_error}{truncation_note}",
                    title="Code Runner",
                    code=code,
                    language="julia",
                    success=False,
                )
            )

        # Return truncated error to agent
        code_runner_error_message = (
            "## Code runner error:\n"
            + "Running the code generated failed with the following error:\n"
            + display_error
        )
        if is_truncated:
            code_runner_error_message += f"\n\n(Error output truncated from {len(julia_error_message):,} to {OUTPUT_TRUNCATION_LIMIT:,} characters)"
        return code_runner_error_message, True

    runtime = round(result["runtime"], 2)
    print_to_console(
        text=f"Code succeeded in {runtime} seconds!",
        title="Code Runner",
        border_style=colorscheme.success,
    )
    # Log success
    if logger:
        logger.log(
            CodeRunnerEntry(
                content=f"Code succeeded in {runtime} seconds!",
                title="Code Runner",
                code=code,
                language="julia",
                success=True,
            )
        )

    return "", False


def check_code(
    state: State,
    config: RunnableConfig,
):
    configuration = BaseConfiguration.from_runnable_config(config)

    code_block = state.code_block
    code = code_block.get_full_code()

    # Return early if no code is present
    if code_block.is_empty():
        return {"error": False}

    messages_list = []
    check_code_bool = True
    user_response = ""
    new_code = code
    if configuration.human_interaction.code_check:
        if cli_mode:
            import jutulgpt.human_in_the_loop.cli as cli

            check_code_bool, user_response, new_code = cli.response_on_check_code(
                code=code
            )
        else:
            import jutulgpt.human_in_the_loop.ui as ui

            check_code_bool, user_response, new_code = ui.response_on_check_code(
                code=code
            )

    # Return early if user provides response or does not want to check code
    if user_response:
        return {"error": True, "messages": [HumanMessage(content=user_response)]}
    if not check_code_bool:
        return {"error": False}

    code_updated = False
    if new_code != code:
        code_updated = True
        code = new_code
        code_update_message = (
            "The code was manually updated to the following. "
            + "This is what will be checked:\n"
            + add_julia_context(code)
        )
        messages_list.append(HumanMessage(content=code_update_message))

    # Then shorten the code for faster simulations
    # code = shorter_simulations(code)

    # Running the linter
    linting_message, linting_issues_found = _run_linter(code, print_code=False)

    # Running the code
    code_running_message, code_running_issues_found = _run_julia_code(
        code, print_code=False
    )

    # If we did not find any issues, we return the final code
    if not linting_issues_found and not code_running_issues_found:
        return {"error": False, "messages": messages_list}

    # If we found issues, we prepare the feedback messages
    feedback_message = "# Code check issues found. Please use these to fix your code:\n"
    if linting_issues_found:
        feedback_message += linting_message + "\n"
    if code_running_issues_found:
        feedback_message += code_running_message

    # Return the feedback messages and and error flag
    messages_list.append(HumanMessage(content=feedback_message))

    # If the code fails, the human can skip the fixing.
    check_code_bool = True
    user_response = ""
    if configuration.human_interaction.fix_error:
        if cli_mode:
            import jutulgpt.human_in_the_loop.cli as cli

            check_code_bool, user_response = cli.response_on_error()
        else:
            import jutulgpt.human_in_the_loop.ui as ui

            check_code_bool, user_response = ui.response_on_error()

    if not check_code_bool:
        messages_list.append(
            HumanMessage(
                content="The code failed, but the user decided to skip fixing it."
            )
        )
        return {"messages": messages_list, "error": False}
    if user_response:
        messages_list.append(HumanMessage(content=user_response))

    if code_updated:
        new_code_block = get_code_from_response(code, within_julia_context=False)
        return {"messages": messages_list, "error": True, "code_block": new_code_block}
    return {"messages": messages_list, "error": True}

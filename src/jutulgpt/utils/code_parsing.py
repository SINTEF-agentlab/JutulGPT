"""Functions for extracting and parsing code from LLM responses."""

import re

from jutulgpt.state import CodeBlock, State


def format_code_response(code: CodeBlock) -> str:
    """
    Format a CodeBlock object as a Markdown Julia code block string.

    Args:
        code (CodeBlock): The code block containing imports and code.

    Returns:
        str: Markdown-formatted Julia code block, or empty string if no code/imports.
    """
    out = ""
    if code.imports != "" or code.code != "":
        out += "```julia\n"
        if code.imports != "":
            out += f"{code.imports}\n\n"
        if code.code != "":
            out += f"{code.code}\n"
        out += "```"
    return out


def add_julia_context(code: str) -> str:
    return f"```julia\n{code}\n```"


def remove_julia_context(code: str) -> str:
    return code.replace("```julia\n", "").replace("\n```", "")


def split_code_into_lines(code: str):
    """
    Split Julia code into blocks based on bracket balance ((), [], {}).
    Multi-line constructs are supported without relying on language-specific keywords.

    Args:
        code (str): Julia code as a string.

    Returns:
        list: List of code blocks as strings, each with balanced brackets.
    """
    lines = code.splitlines()
    blocks = []
    current_block = []
    parens = brackets = braces = 0

    for line in lines:
        stripped = line.strip()
        if not stripped and not current_block:
            continue  # Skip empty lines outside a block

        # Update bracket counts
        parens += line.count("(") - line.count(")")
        brackets += line.count("[") - line.count("]")
        braces += line.count("{") - line.count("}")

        current_block.append(line)

        # If all brackets are balanced, this is a complete block
        if parens == 0 and brackets == 0 and braces == 0:
            blocks.append("\n".join(current_block))
            current_block = []

    # In case something is left unbalanced (e.g., trailing incomplete block)
    if current_block:
        blocks.append("\n".join(current_block))

    return blocks


def _get_code_string_from_response(response: str) -> str:
    """
    Extract Julia code from one or more Markdown-style Julia code blocks in a response string.
    If multiple code blocks are found, they are joined in chronological order.

    Args:
        response (str): The response string containing Markdown Julia code block(s).

    Returns:
        str: The extracted Julia code (joined if multiple blocks), or an empty string if not found.
    """
    if not response:
        return ""
    matches = re.findall(r"```julia\s*([\s\S]*?)```", response, re.IGNORECASE)
    if matches:
        # Join multiple code blocks with double newlines to separate them
        code_blocks = [match.strip() for match in matches if match.strip()]
        return "\n\n".join(code_blocks)
    return ""


def get_code_from_response(
    response: str, within_julia_context: bool = True
) -> CodeBlock:
    """
    Extract Julia code and import statements from a Markdown code block in a response string.

    Args:
        response (str): The response string containing a Markdown Julia code block.
        within_julia_context (bool): If True, assumes the response is within a Julia context. If False, assumes the entire response is code.

    Returns:
        CodeBlock: An object containing separated imports and code.
    """
    code_str = (
        _get_code_string_from_response(response) if within_julia_context else response
    )

    if not code_str:
        return CodeBlock(imports="", code="")

    import_lines = []
    code_lines = []
    for line in code_str.splitlines():
        if line.strip().startswith(("using ")):
            import_lines.append(line.strip())
        else:
            code_lines.append(line)

    return CodeBlock(
        imports="\n".join(import_lines), code="\n".join(code_lines).strip()
    )


def get_last_code_response(state: State) -> CodeBlock:
    """
    Get the last AI-generated code response from the state as a CodeBlock.

    Args:
        state (State): The current State object containing messages.

    Returns:
        CodeBlock: The extracted code block from the last AI message, or empty if not found.
    """
    from jutulgpt.utils.model import get_message_text

    last_message = state.messages[-1]

    # Include the human in case the human-in-the-loop updates the generated code.

    # Always use normalized string view of messages for parsing.
    if last_message.type == "ai" or last_message.type == "human":
        last_message_content = get_message_text(last_message)
    else:
        last_message_content = ""
    code_block = get_code_from_response(last_message_content)
    return code_block

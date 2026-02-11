from jutulgpt.cli import colorscheme, print_to_console
from jutulgpt.julia.julia_code_runner import run_julia_file


def _parse_julia_doc_output(output: str):
    """
    Parse the output from julia_get_function_documentation.jl to extract function names and documentation.
    """
    lines = output.splitlines()
    func_names = []
    documentation = ""

    # Find the indices for the markers
    try:
        fn_start = lines.index("FUNCTION NAMES:") + 1
        doc_start = lines.index("DOCUMENTATION") + 1
    except ValueError:
        return [], ""

    # Function names are printed as a Julia array, e.g. ["foo", "bar"]
    # Join lines between fn_start and doc_start-1
    func_names_raw = "".join(lines[fn_start : doc_start - 1]).strip()
    # Remove brackets and quotes, split by comma
    if func_names_raw.startswith("[") and func_names_raw.endswith("]"):
        func_names_raw = func_names_raw[1:-1]
    func_names = [
        name.strip().strip('"') for name in func_names_raw.split(",") if name.strip()
    ]

    # Documentation is everything after doc_start
    documentation = "\n".join(lines[doc_start:]).strip()

    return func_names, documentation


def get_function_documentation(code: str) -> tuple[list[str], str]:
    """
    Returns:
        list[str]: A list of function names found in the code.
        str: The documentation string for the functions.
    """

    try:
        res, err, returncode = run_julia_file(
            code=code, julia_file_name="julia_get_function_documentation.jl"
        )

        if returncode != 0:
            raise RuntimeError(f"Julia documentation script failed: {err}")

        func_names, documentation = _parse_julia_doc_output(res)

        if func_names:
            func_names = [
                name for name in func_names if name != "String[]"
            ]  # filter out empty function names
            out_text = "Retrieved functions: " + ", ".join(func_names)
            # print_to_console(
            #     text=out_text,
            #     title="Function Documentation Retriever",
            #     border_style=colorscheme.success,
            # )
        else:
            print_to_console(
                text="No function documentation found!",
                title="Function Documentation Retriever",
                border_style=colorscheme.error,
            )

        return func_names, documentation

    except Exception as e:
        print_to_console(
            text="Error retrieving function documentation: " + str(e),
            title="Function Documentation Retriever",
            border_style=colorscheme.error,
        )
        return [], ""


def get_function_documentation_from_list_of_funcs(
    func_names: list[str],
) -> tuple[list[str], str]:
    """
    Get function documentation from a list of function names.

    Args:
        funcs (list[str]): List of function names to get documentation for.

    Returns:
        tuple[list[str], str]: A tuple containing a list of function names and their documentation.
    """
    code = "\n".join(f"{func_name}();" for func_name in func_names)

    print_to_console(
        text="Retrieving documentation for functions: " + ", ".join(func_names),
        title="Function Documentation Retriever",
        border_style=colorscheme.message,
    )
    return get_function_documentation(code)

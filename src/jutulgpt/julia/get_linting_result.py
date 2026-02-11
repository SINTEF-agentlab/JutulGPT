from jutulgpt.cli import colorscheme, print_to_console
from jutulgpt.julia.julia_code_runner import run_julia_file


def get_linting_result(code: str) -> str:
    try:
        res, err, returncode = run_julia_file(code=code, julia_file_name="julia_lint_script.jl")

        lines = res.splitlines()
        for i, line in enumerate(lines):
            if "STARTING LINT:" in line:
                linting_result = "\n".join(lines[i + 1 :])

                if linting_result:
                    print_to_console(
                        text=linting_result,
                        title="Linter Result",
                        border_style=colorscheme.error,
                    )
                else:
                    print_to_console(
                        text="No linting issues found!",
                        title="Linter Result",
                        border_style=colorscheme.success,
                    )
                return linting_result

        return ""

    except Exception as _:
        return ""

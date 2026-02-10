"""Functions for transforming Julia code before execution."""

import re
from typing import List

from jutulgpt.state import CodeBlock
from jutulgpt.utils.code_parsing import split_code_into_lines


def check_for_package_install(code_block: CodeBlock) -> bool:
    not_allowed = [
        "using Pkg",  # Pkg is used to install packages, which is not allowed
        "Pkg.add",  # Pkg.add is used to install packages, which is not allowed
        "Pkg.update",  # Pkg.update is used to update packages, which is not allowed
        "Pkg.instantiate",  # Pkg.instantiate is used to install dependencies, which is not allowed
    ]
    if any(item in code_block.imports for item in not_allowed):
        return True
    if any(item in code_block.code for item in not_allowed):
        return True
    return False


def remove_plotting(code: str) -> str:
    """
    Remove GLMakie usage and plotting code from Julia code blocks.
    F.ex:
    - Removes 'using GLMakie' or 'using ... GLMakie ...' from using statements.
    - Removes lines that define or use 'fig', 'ax', or call 'lines!'.
    - Removes lines that are just 'fig' (returning the figure).
    """
    # TODO: This is a very naive implementation, it should be improved.
    remove_functions = [
        "fig",
        "plt",
        "ax",
        "scatter",
        "Colorbar",
        "Axis",
        "lines",
        "plot_reservoir",
        "plot_well_results",
        "plot_reservoir_measurables",
        "plot_reservoir_simulation_result",
        "plot_well!",
        "myplot",
        "plot_cell_data",
        "plot_mesh_edges",
        "plot_mesh",
        "plot_co2_inventory",
        "println",  # To avoid printing to terminal
    ]

    # lines = code.splitlines()
    lines = split_code_into_lines(code)
    new_lines = []
    for line in lines:
        stripped = line.strip()
        # Remove 'using GLMakie' or 'using ... GLMakie ...'
        if stripped.startswith("using"):
            # Remove 'GLMakie' from the using statement
            # Handles cases like: using JutulDarcy, GLMakie, Jutul;
            # and: using GLMakie
            line = re.sub(r",?\s*GLMakie,?", "", line)
            # Remove trailing/leading commas and extra spaces
            line = re.sub(r"using\s*,", "using ", line)
            line = re.sub(r",\s*;", ";", line)
            # If nothing left after 'using', skip the line
            if re.match(r"^\s*using\s*;?\s*$", line):
                continue
            # If the line is now empty, skip it
            if not line.strip():
                continue
        # Remove lines that define or use fig, ax, or call lines!
        if stripped == "fig":
            continue

        if any(func in stripped for func in remove_functions):
            continue
        new_lines.append(line)
    return "\n".join(new_lines)


def shorten_first_argument(code: str, simulation_functions: List[str]) -> str:
    """
    For each simulation function, append '[1:1]' to the first argument of the function call.
    E.g., sim_func_1(foo, bar) -> sim_func_1(foo[1:1], bar)
    """

    for func in simulation_functions:
        # Match the function call and capture the first argument
        pattern = rf"({func}\s*\()\s*([^,)\s]+)(.*?\))"

        def replacer(match):
            before = match.group(1)
            first_arg = match.group(2)
            after = match.group(3)
            return f"{before}{first_arg}[1:1]{after}"

        code = re.sub(pattern, replacer, code, flags=re.DOTALL)

    return code


def replace_case(code: str, case_name: str, simulation_functions: List[str]) -> str:
    """
    Replacing the 'case' with 'case[1:1]'.
    """
    for func in simulation_functions:
        # Match 'case' as a whole word in the argument list
        pattern = rf"({func}\s*\(.*?)(\b{case_name}\b)(.*?\))"

        def replacer(match):
            before = match.group(1)
            after = match.group(3)
            return f"{before}{case_name}[1:1]{after}"

        code = re.sub(pattern, replacer, code, flags=re.DOTALL)
    return code


def shorter_simulations(code: str) -> str:
    """
    In the case when some simulation is called, this function replaces it with a shorter simulation.
    """
    simulation_functions = [
        "simulate_reservoir",
    ]
    original_code = code

    code = replace_case(
        code=code, case_name="case", simulation_functions=simulation_functions
    )
    code = replace_case(
        code=code, case_name="dt", simulation_functions=simulation_functions
    )
    code = replace_case(
        code=code, case_name="timesteps", simulation_functions=simulation_functions
    )

    if original_code == code:
        # WARNING: If the previous functions have not changed the code, it might inidate that some other name is used. However, this might we a weird assumption?
        code = shorten_first_argument(
            code=code, simulation_functions=simulation_functions
        )

    return code

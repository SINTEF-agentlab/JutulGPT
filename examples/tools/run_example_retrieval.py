"""
Example of how can invoke a tool for testing how it works.

The tools you can test are:
execute_terminal_command
run_julia_code
run_julia_linter
get_working_directory
list_files_in_directory
read_from_file
write_to_file
grep_search
retrieve_function_documentation
retrieve_jutuldarcy_examples

We use the grep_search as an example here.

"""

from jutulgpt.tools import retrieve_jutuldarcy_examples

if __name__ == "__main__":
    query = "CartesianMesh"
    out = retrieve_jutuldarcy_examples.invoke({"query": query})
    print(out)

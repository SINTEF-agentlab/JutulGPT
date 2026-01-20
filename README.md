# JutulGPT

An AI assistant for JutulDarcy!

![CLI example](media/JutulGPT_CLI.png "CLI example")

## Getting started

### Prerequisites

This project requires both **Python** and **Julia**, along with some system-level dependencies. Make sure these are installed:

- `git`: See [git downloads](https://git-scm.com/downloads).
- `Python3 >=3.12`: See NOTE or [Download Python](https://www.python.org/downloads/)
- `Julia`: Package tested on version 1.11.6. See [Installing Julia](https://julialang.org/install/).
- `build-essential`
- `graphviz` and `graphviz-dev`: See [Graphviz download](https://graphviz.org/download/)

Optional:

- `uv`: Recommended package manager. See [Installing uv](https://docs.astral.sh/uv/getting-started/installation/).
- `ollama`: For running local models. See [Download Ollama](https://ollama.com/download).

> NOTE: See [Installing python](https://docs.astral.sh/uv/guides/install-python/) for installing Python using `uv`.

### Step 1: Python

Retireve the code by cloning the repository

```bash
# Clone and choose the repo
git clone https://github.com/ellingsvee/JutulGPT.git
cd JutulGPT/
```

If you are using `uv`, initialize the environment by

```bash
# Initialize the enviroment
uv venv
source .venv/bin/activate

# Install packages
uv sync
```

If encountering an error due to the `pygraphviz` package, try explicitly installing it using a package manager.

For MacOS:
```bash
# Note: This example is for MacOS using Homebrew. Adjust accordingly for your OS/package manager.
brew install graphviz
uv add --config-settings="--global-option=build_ext" \
            --config-settings="--global-option=-I$(brew --prefix graphviz)/include/" \
            --config-settings="--global-option=-L$(brew --prefix graphviz)/lib/" \
            pygraphviz
```
For Ubuntu:
```bash
sudo apt install libgraphviz-dev graphviz
```

### Step 2: Julia

For running Julia code we also need to set up a working Julia project.

```bash
julia
# In Julia
julia> import Pkg; Pkg.activate("."); Pkg.instantiate()
```

This will install all the necessary packages listed in the `Project.toml` the first time you invoke the agent.

### Step 3: Setup environment

You then have to set the environment variables. Generate a `.env` file by

```bash
cp .env.example .env
```

and modify it by providing your own `OPENAI_API_KEY` key.  For running in the UI you also must provide an `LANGSMITH_API_KEY` key.

### Step 4: Test it

Finally, try to initialize the agent by

```bash
uv run examples/agent.py
```

This should install the necessary Julia packages before running. You might need to re-run the model after the installation.

## Basic usage

Two different agents are implemented.

### `Agent`

The first agent follows an evaluator-optimizer workflow, where code is first generated and then evaluated. This strategy works well for smaller models and more specific tasks. It is f.ex. suggested to use this model for generating code to set up a simulation.

![Evaluator Optimizer](media/Evaluator_optimizer.png "Evaluator Optimizer")

Run the agent in the CLI by

```bash
uv run examples/agent.py
```

### `Autonomous Agent`

The second agent has more available tools, and can interact with the environment in a more sophisticated way. For sufficiently large LLMs, this agent can provide a more _Copilot_-like experience.

![Autonomous Agent](media/Autonomous_Agent.png "Autonomous Agent")

Run the agent in the CLI by

```bash
uv run examples/autonomous_agent.py
```

## Settings and configuration

The agent is configured in the `src/jutulgpt/configuration.py` file.  

### Model selection (recommended: CLI flag)

You can select the model at startup with `--model`. The default is:

- `--model gpt-5.2-reasoning` (alias for the `gpt-5.2-reasoning` preset)

Examples:

```bash
uv run examples/agent.py --model gpt-4.1
uv run examples/autonomous_agent.py --model gpt-5.2-reasoning
uv run examples/autonomous_agent.py --model qwen3:14b-thinking
```

### Model configuration (presets)

Model presets are defined as `ModelConfig` constants in `src/jutulgpt/configuration.py`:

- `provider` / `model`
- `context_window` (used by context tracking/summarization)
- `llm_kwargs` (provider-specific kwargs forwarded to LangChain `init_chat_model(...)`)

See `docs/model_configuration.md` for the full table and details.

More advanced settings are set in the `BaseConfiguration`. LangGraph will turn these into a `RunnableConfig`, which enables easier configuration at runtime.  You specify the following settings:

- `human_interaction`: Enable human-in-the-loop. See the `HumanInteraction` class in the configuration file for detailed control.
- `embedding_model`: Name of the embedding model to use. By default equal to the `EMBEDDING_MODEL_NAME`.
- `retriever_provider`: The vector store provider to use for retrieval.
- `examples_search_type`: Defines the type of search that the retriever should perform when retrieving examples.
- `examples_search_kwargs`: Keyword arguments to pass to the search function of the retriever when retrieving examples. See [LangGraph documentation](https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.as_retriever) for details about what arguments works for the different search types.
- `rerank_provider`: The provider user for reranking the retrieved documents.
- `rerank_kwargs`: Keyword arguments provided to the reranker.
- `agent_model`: The language model used for generating responses. Should be in the form `provider:model`. Defaults to the active preset (see `ACTIVE_MODEL`).
- `autonomous_agent_model`: See `agent_model`.
- `agent_prompt`: The prompt used for the agent.
- `autonomous_agent_prompt`: The prompt used for the autonomous agent.

The settings can be specified by passing a configuration dictionary when invoking the models. See f.ex the `run()` function in `src/jutulgpt/agents/agent_base.py`. Alternatively, the GUI provides a custom interface where the settings can be selected.

## Interfaces

### CLI

Enable the CLI-mode by in `src/jutulgpt/configuration.py` setting

```python
cli_mode = True
```

This gives you a nice interface for asking questions, retrieving info, generating and running code etc. Both agents can also read and write to files.

### VSCode integration using MCP

For calling using JutulGPT from VSCode, it can communicate with Copilot through setting up an [MCP server](https://code.visualstudio.com/docs/copilot/customization/mcp-servers).

To enable MCP server in JutulGPT, in `src/jutulgpt/configuration.py` set

```python
cli_mode = False # Disable CLI mode
mcp_mode = True
```

and start JutulGPT through the [Langgraph CLI](https://docs.langchain.com/langsmith/cli) by running

```bash
source .venv/bin/activate # If not already activated
langgraph dev # Starts local dev server
```

Then, in the VSCode workspace where you want to use JutulGPT, add the an MCP server through a `mcp.json` file. See the `.vscode.example/mcp.json` file for an example. Finally, select the JutulGPT MCP as a tool in the Copilot settinsgs. See [Use MCP tools in chat](https://code.visualstudio.com/docs/copilot/customization/mcp-servers#_use-mcp-tools-in-chat) for how ot do this!

### GUI

![GUI example](media/JutulGPT_GUI.png "GUI example")

The JutulGPT also has an associated GUI called [JutulGPT-GUI](https://github.com/ellingsvee/JutulGPT-GUI).  For using the GUI, you must disable the CLI-mode. To this by setting `cli_mode = False` in `src/jutulgpt/configuration.py`.

Install it by following the instructions in the repository. Alternatively do

```bash
cd .. # Move to parent directory
git clone https://github.com/ellingsvee/JutulGPT-GUI.git # Clone JutulGPT-GUI
cd JutulGPT-GUI/
pnpm install
cd ../JutulGPT/ # Move back to JutulGPT
```

To run the GUI locally, you have to use the [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/) tool. Start it by

```bash
langgraph dev # Run from JutulGPT/ directory
```

and start the GUI from the JutulGPT-GUI directory by running

```bash
pnpm dev # Run from JutulGPT-GUI/ directory
```

The GUI can now be accessed on `http://localhost:3000/` (or some other location depending on your JutulGPT-GUI configuration).

> NOTE: Remember to set `cli_mode = False` in `src/jutulgpt/configuration.py`.

## Fimbul (WARNING)

There is some legacy code for generating code for the Fimbul package. I have removed a lot of it, but it can be re-implemented by adding some tools and modifying the prompts. My suggestion is to get familiar with the current tools fot JutulDarcy, and then later extend to Fimbul.

## Testing

Tests are set up to be implemented using [pytest](https://docs.pytest.org/en/stable/). They can be written in the `tests/` directory. Run by the command

```bash
uv run pytest
```

> Note: No tests have yet been implemented.

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

JutulGPT is configured through the **`jutulgpt.toml`** file in the project root. Edit this file to change the model, retrieval settings, context thresholds, logging, and more — no Python editing required.

**Priority order:** Python defaults < `jutulgpt.toml` < CLI `--model` flag < LangGraph runtime config

If `jutulgpt.toml` is missing, all settings fall back to sensible Python defaults.

### `jutulgpt.toml` sections

| Section | What it controls |
|---|---|
| `[mode]` | CLI vs MCP mode (`cli = true` / `mcp = true`) |
| `[model]` | Default model preset and reasoning summary display |
| `[retrieval]` | Retriever provider, search type, search kwargs, reranking |
| `[human_interaction]` | Which steps require human approval |
| `[display]` | Console/log output truncation |
| `[context]` | Summarization/trim thresholds, recursion limit |
| `[output]` | Output and summary message truncation limits |
| `[logging]` | Log file settings (enabled, directory, prefix) |

Example — switch the default model to GPT-4.1:

```toml
[model]
preset = "gpt-4.1"
```

Example — use FAISS retrieval with similarity search:

```toml
[retrieval]
provider = "faiss"
search_type = "similarity"

[retrieval.search_kwargs]
k = 5
```

### Model selection (recommended: CLI flag)

You can override the TOML model preset at startup with `--model`:

```bash
uv run examples/agent.py --model gpt-4.1
uv run examples/autonomous_agent.py --model gpt-5.2-reasoning
uv run examples/autonomous_agent.py --model qwen3:14b-thinking
```

If `--model` is omitted, the value from `[model].preset` in `jutulgpt.toml` is used (default: `gpt-5.2-reasoning`).

### Model configuration (presets)

Model presets are defined as `ModelConfig` constants in `src/jutulgpt/configuration.py`:

- `provider` / `model`
- `context_window` (used by context tracking/summarization)
- `llm_kwargs` (provider-specific kwargs forwarded to LangChain `init_chat_model(...)`)

See `docs/model_configuration.md` for the full table and details.

### Advanced runtime settings

More advanced settings are available in the `BaseConfiguration` dataclass. LangGraph turns these into a `RunnableConfig` for runtime configuration. Most of these read their defaults from `jutulgpt.toml` automatically. See `src/jutulgpt/configuration.py` for the full list.

## Interfaces

### CLI

CLI mode is enabled by default. You can also explicitly set it in `jutulgpt.toml`:

```toml
[mode]
cli = true
mcp = false
```

This gives you a nice interface for asking questions, retrieving info, generating and running code etc. Both agents can also read and write to files.

### VSCode integration using MCP

For calling using JutulGPT from VSCode, it can communicate with Copilot through setting up an [MCP server](https://code.visualstudio.com/docs/copilot/customization/mcp-servers).

To enable MCP server in JutulGPT, set the following in `jutulgpt.toml`:

```toml
[mode]
cli = false
mcp = true
```

and start JutulGPT through the [Langgraph CLI](https://docs.langchain.com/langsmith/cli) by running

```bash
source .venv/bin/activate # If not already activated
langgraph dev # Starts local dev server
```

Then, in the VSCode workspace where you want to use JutulGPT, add the an MCP server through a `mcp.json` file. See the `.vscode.example/mcp.json` file for an example. Finally, select the JutulGPT MCP as a tool in the Copilot settinsgs. See [Use MCP tools in chat](https://code.visualstudio.com/docs/copilot/customization/mcp-servers#_use-mcp-tools-in-chat) for how ot do this!

### GUI

![GUI example](media/JutulGPT_GUI.png "GUI example")

The JutulGPT also has an associated GUI called [JutulGPT-GUI](https://github.com/ellingsvee/JutulGPT-GUI).  For using the GUI, you must disable the CLI-mode. Do this by setting `cli = false` under `[mode]` in `jutulgpt.toml`.

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

> NOTE: Remember to set `cli = false` under `[mode]` in `jutulgpt.toml`.

## Fimbul (WARNING)

There is some legacy code for generating code for the Fimbul package. I have removed a lot of it, but it can be re-implemented by adding some tools and modifying the prompts. My suggestion is to get familiar with the current tools fot JutulDarcy, and then later extend to Fimbul.

## Testing

Tests are set up to be implemented using [pytest](https://docs.pytest.org/en/stable/). They can be written in the `tests/` directory. Run by the command

```bash
uv run pytest
```

> Note: No tests have yet been implemented.

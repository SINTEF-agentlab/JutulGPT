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

If encountering an error due to the `pygraphviz` package, try explicitly installing it using
```bash
# Note: This example is for MacOS using Homebrew. Adjust accordingly for your OS/package manager.
brew install graphviz
uv add --config-settings="--global-option=build_ext" \
            --config-settings="--global-option=-I$(brew --prefix graphviz)/include/" \
            --config-settings="--global-option=-L$(brew --prefix graphviz)/lib/" \
            pygraphviz
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

The agent is configured in the `jutulgpt.toml` file. This is where you specify things like which LLM to use, and the number of examples to retrieve.

> Note: We are in the process of migrating settings from the `src/jutulgpt/configuration.py` file, so some settings might still be there.


## Interfaces

### CLI

Enable the CLI-mode by in `jutulgpt.toml` by setting `cli_mode = True`. Then, you can run the agents through the CLI by running the examples as shown above. This gives you a nice interface for asking questions, retrieving info, generating and running code etc. Both agents can also read and write to files.

### VSCode integration using MCP

For calling using JutulGPT from VSCode, it can communicate with Copilot through setting up an [MCP server](https://code.visualstudio.com/docs/copilot/customization/mcp-servers).

To enable MCP server in JutulGPT, specify in `jutulgpt.toml`,  and start JutulGPT through the [Langgraph CLI](https://docs.langchain.com/langsmith/cli) by running

```bash
source .venv/bin/activate # If not already activated
langgraph dev # Starts local dev server
```

Then, in the VSCode workspace where you want to use JutulGPT, add the an MCP server through a `mcp.json` file. See the `.vscode.example/mcp.json` file for an example. Finally, select the JutulGPT MCP as a tool in the Copilot settinsgs. See [Use MCP tools in chat](https://code.visualstudio.com/docs/copilot/customization/mcp-servers#_use-mcp-tools-in-chat) for how ot do this!

### GUI

![GUI example](media/JutulGPT_GUI.png "GUI example")

The JutulGPT also has an associated GUI called [JutulGPT-GUI](https://github.com/ellingsvee/JutulGPT-GUI).  For using the GUI, you must disable the CLI-mode. To this by setting `cli = False` in `jutulgpt.toml`.

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

> NOTE: Remember to set `cli = False` in `jutulgpt.toml`.

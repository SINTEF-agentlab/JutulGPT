# Model configuration

Model selection is configured in `src/jutulgpt/configuration.py` via `ModelConfig` presets.

JutulGPT currently supports these providers:

- **OpenAI**: via API key (`OPENAI_API_KEY`)
- **Ollama**: local models

## Supported models

`ACTIVE_MODEL_CONFIG` selects one of the presets below.

## CLI model selection

When running via the CLI, you can select a preset at startup:

```bash
python examples/agent.py --model gpt-5.1-mini-reasoning
python examples/autonomous_agent.py --model gpt-5.1-reasoning
python examples/autonomous_agent.py --model qwen3:14b-thinking
```

Default:

- `--model gpt-5.1-mini-reasoning` (alias for `gpt-5-mini-reasoning`)

| Preset name (`configuration.py`) | Provider | Model | Context window | Reasoning | Verbosity | Reasoning effort | Reasoning summary |
|---|---|---:|---:|---|---|---|---|
| `OPENAI_GPT_4_1` | OpenAI | `gpt-4.1` | 200k | No | `medium` | `medium` | — |
| `OPENAI_GPT_5_1` | OpenAI | `gpt-5.1` | 200k | No | `low` | `medium` | — |
| `OPENAI_GPT_5_1_REASONING` | OpenAI | `gpt-5.1` | 200k | Yes | `low` | `medium` | `auto` |
| `OPENAI_GPT_5_MINI` | OpenAI | `gpt-5-mini` | 200k | No | `low` | `medium` | — |
| `OPENAI_GPT_5_MINI_REASONING` | OpenAI | `gpt-5-mini` | 200k | Yes | `low` | `medium` | `auto` |
| `OLLAMA_QWEN3_14B_THINKING` | Ollama | `qwen3:14b` | 32k | Yes | — | — | — |
| `OLLAMA_QWEN3_14B` | Ollama | `qwen3:14b` | 32k | No | — | — | — |

## OpenAI: Responses API

OpenAI presets use the **OpenAI Responses API** (`use_responses_api=True`). This is also how **reasoning summary blocks** are returned for reasoning-enabled presets.

## Important settings

### `context_window`

The model context window size in tokens. This is used for context tracking and the summarization thresholds.

### `llm_kwargs`

Provider-specific keyword arguments forwarded to LangChain `init_chat_model(...)`.

Common examples:

- **OpenAI**:
  - `use_responses_api`: `true`
  - `verbosity`: `"low" | "medium" | "high"`
  - `reasoning`: `{ "effort": "low|medium|high", "summary": "auto|concise|detailed" }`
- **Ollama**:
  - `reasoning`: `true` (for “thinking” models where this suppresses thoughts)
  - `num_ctx`: context window size (important for long prompts; Ollama defaults to a small value if not set)
  - `temperature`: for Qwen3, a common starting point is ~0.6 (thinking) / ~0.7 (non-thinking)

### `verbosity` (OpenAI via `llm_kwargs`)

Controls response verbosity for supported OpenAI models:

- `"low"`: shorter, more direct answers
- `"medium"`: balanced detail (default for several presets)
- `"high"`: more verbose answers

Note: supported values are **model-dependent** (e.g. `gpt-4.1` currently only supports `"medium"`).

### `reasoning.effort` (OpenAI via `llm_kwargs`)

Controls how much compute the model should spend reasoning (reasoning-enabled presets only):

- `"low"` / `"medium"` / `"high"`

Higher effort is typically slower and more expensive.

### `reasoning.summary` (OpenAI via `llm_kwargs`)

Controls how the model returns a *summary* of its reasoning (reasoning-enabled presets only):

- `"auto"`: model chooses
- `"concise"`: shorter summary
- `"detailed"`: longer summary

The summary is what JutulGPT can display when `SHOW_REASONING_SUMMARY=True`.

## Reasoning summaries (display)

- **Toggle**: `SHOW_REASONING_SUMMARY = True`
- When enabled and the OpenAI model returns reasoning summary blocks, JutulGPT prints them in the CLI and stores them in the session log.

### Access requirement

Some OpenAI reasoning features/models require **organization verification**. If you don’t have access, requests may fail.

- OpenAI org settings: [Organization settings](https://platform.openai.com/settings/organization)

## Adding a new model preset

1. **Add a new `ModelConfig`** constant in `src/jutulgpt/configuration.py` (copy an existing preset).
2. **Set provider + model**: `provider="openai"|"ollama"`, `model="..."`
3. **Set `context_window`** (default is 200k; Qwen is 32k in current presets).
4. **Set `llm_kwargs`** for provider-specific options:
   - OpenAI: include `{"use_responses_api": True, "verbosity": "low", ...}` and optional `reasoning` dict.
   - Ollama: include `{"num_ctx": <tokens>}` to increase context, and optionally `{"reasoning": True}` for thinking models.
5. **Select it** by setting `ACTIVE_MODEL_CONFIG = YOUR_NEW_PRESET`.

## Environment setup

### Ollama

Ollama models must be available locally. If you see errors like “model not found, try pulling it first”, run:

```bash
ollama pull <model>
```

### OpenAI

```bash
export OPENAI_API_KEY="your-api-key"
```

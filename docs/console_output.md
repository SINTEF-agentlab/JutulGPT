# Console Output

JutulGPT uses [Rich](https://rich.readthedocs.io/) for terminal rendering: panels, markdown rendering, and live streaming updates.

## Streaming output

During an LLM call, `stream_to_console()` renders a live panel that updates as tokens arrive.

- **Reasoning Summary** (yellow panel): shown when `SHOW_REASONING_SUMMARY=True` and the model returns reasoning summary blocks (OpenAI Responses API)
- **Agent Output** (cyan panel): streamed response text with a character count

Note: reasoning summaries only appear when the selected model/preset requests them (e.g. OpenAI reasoning presets via `llm_kwargs["reasoning"]`) *and* `SHOW_REASONING_SUMMARY=True`.

To avoid Rich `Live` corruption from overly large renders, streaming uses a **tail view** (last N lines). When streaming completes, the **full** response is printed to scrollback (optionally rendered as Markdown).

### Phase behavior

- **Reasoning phase**: show reasoning summary tail
- **Text phase**: show agent output tail
- **Transition**: when text begins, the reasoning summary is printed once to scrollback
- **End**: full agent output is printed to scrollback

## Limitations

### Terminal resize during streaming

Resizing the terminal while Rich `Live` is active may cause display artifacts. This is a known limitation of terminal cursor-positioned rendering.

- **Workaround**: avoid resizing during streaming; final scrollback output will still render correctly.

## Entry points

- `stream_to_console()`: live streaming + final scrollback printing
- `print_to_console()`: one-shot panel printing

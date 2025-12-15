# Context Management

How JutulGPT manages LLM context for long conversations.

## Context Budget

```
┌─────────────────── MODEL_CONTEXT_WINDOW (200k) ───────────────────┐
│                                                                   │
│  System (static, counted once):                                   │
│  ├── System prompt + workspace info                               │
│  └── Tool definitions                                             │
│                                                                   │
│  Conversation (dynamic):                                          │
│  ├── Context summary (when active)                                │
│  ├── Tool output (ToolMessages + tool_calls)                      │
│  └── Messages (HumanMessage + AIMessage content)                  │
│                                                                   │
│  Thresholds:                                                      │
│  ├── 70% (140k) → Summarization triggered                         │
│  └── 90% (180k) → Safety trim applied                             │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

| Constant | Default | Purpose |
|----------|---------|---------|
| `MODEL_CONTEXT_WINDOW` | 200k tokens | Total context budget |
| `CONTEXT_USAGE_THRESHOLD` | 0.7 | Trigger summarization at 70% |
| `CONTEXT_TRIM_THRESHOLD` | 0.9 | Safety trim at 90% (fallback) |
| `CONTEXT_DISPLAY_THRESHOLD` | 0.3 | Show usage display above 30% |
| `RECENT_MESSAGES_TO_KEEP` | 10 | Messages preserved during summarization |
| `OUTPUT_TRUNCATION_LIMIT` | 8k chars | Max per tool output |
| `SUMMARY_MSG_LIMIT` | 1k chars | Max per message when summarizing |

## Summarization

When context usage exceeds `CONTEXT_USAGE_THRESHOLD`, older messages are summarized and removed:

```
Before (60 messages, 72% usage):
[System] + [M1, M2, ..., M50, M51, ..., M60]
           └─────────────┘  └────────────┘
             Summarize       Keep (recent 10)

After:
[System] + [Summary] + [M51, ..., M60]
```

When context grows again, re-summarization incorporates the previous summary:

```
First summarization:
  Summarize: [M1, ..., M50]
  Result: Summary1

Second summarization (context grows again):
  Input: [M51, ..., M120] + previous_summary=Summary1
  Result: Summary2 (incorporates Summary1)
```

## Context Usage Display

Shown before user input when usage exceeds `CONTEXT_DISPLAY_THRESHOLD`:

```
                    Context Usage
  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 60.9k/200.0k (30.4%)
  ├─ System:          1.8k (0.9%)     ← prompt + workspace + tool defs
  ├─ Context summary: 485 (0.2%)     ← shown only when active
  ├─ Tool output:     48.6k (24.3%)  ← ToolMessages + tool_calls
  ├─ Messages:        9.9k (5.0%)    ← Human + AI conversation
  └─ Free space:      139.1k (69.6%)
```

The "Context summary" line only appears when summarization has occurred.

# support-bot

A rule-based customer support agent built with [LangGraph](https://github.com/langchain-ai/langgraph). No API key, no LLM, no frontend — just Python, running in your terminal.

It routes each message through a graph: classify the intent by keyword, then hand off to the matching node (order status, refund, FAQ, escalation, or fallback).

## Requirements

- Python 3.10+
- `langgraph`

## Setup

```bash
pip install langgraph
```

## Run

```bash
python support_bot.py
```

Type a message and hit enter. Type `quit` or `exit` to stop.

```
You: where's my order 1001?
Bot: Order 1001: Shipped, arriving Sep 25

You: what's your return policy
Bot: You can return items within 30 days of delivery.

You: this is terrible, get me a human
Bot: I'm connecting you with a human support agent who can help further. Please hold on — someone will be with you shortly.
```

## How it works

```
classify_intent
      │
      ├── order_status  → looks up order ID in FAKE_ORDERS
      ├── refund         → returns refund policy info
      ├── faq             → keyword match against FAQ dict
      ├── escalate       → hands off to a human
      └── other           → fallback message
```

- **`classify_intent`** scans the message for keyword groups (`escalate_words`, `order_words`, `refund_words`, FAQ keys) and picks the first matching intent.
- Each intent has its own node that returns a canned or looked-up reply.
- **`MemorySaver`** keeps the conversation history per `thread_id`, so context persists across turns in a session (in-memory only — it resets when the script stops).

## Customizing

| Want to... | Edit... |
|---|---|
| Add order data | `FAKE_ORDERS` dict |
| Add FAQ answers | `FAQ` dict |
| Change what triggers escalation | `escalate_words` list in `classify_intent` |
| Connect to a real order/refund system | Replace `lookup_order_status()` with a real API/DB call |
| Add a new intent | Add a node, a routing branch, and an edge to `END` in `build_graph()` |

## Known limitations

- Intent detection is keyword-based — phrasing it doesn't recognize falls into `other`.
- Order and refund data are hardcoded/fake; there's no real backend.
- No frontend — it's a terminal loop, meant as a starting point for a graph-based agent, not a production bot.

## Next steps (optional)

- Swap the rule-based nodes for an LLM (Claude, local Ollama model, etc.) for more natural understanding.
- Persist `MemorySaver` to a real database (Postgres, Redis) so history survives restarts.
- Wrap it in a simple API (FastAPI) or UI (Streamlit) if you outgrow the terminal.

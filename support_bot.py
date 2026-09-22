"""
LangGraph Customer Support Bot (Rule-Based, No AI / No API)
-------------------------------------------------------------
Same graph shape as an LLM-powered agent (intent routing, tools,
escalation, memory) but every "brain" is plain Python logic —
keyword matching and if/else. No API key, no model download,
nothing to install except langgraph itself.

Install:
    pip install langgraph

Run:
    python support_bot.py
"""

from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# 1. State
# ---------------------------------------------------------------------------
class SupportState(TypedDict):
    messages: Annotated[list, add_messages]   # list of {"role": ..., "content": ...} dicts
    intent: str
    escalated: bool


# ---------------------------------------------------------------------------
# 2. "Tools" — plain functions, replace with real DB/API calls later
# ---------------------------------------------------------------------------
FAKE_ORDERS = {
    "1001": "Shipped, arriving Sep 25",
    "1002": "Processing, not yet shipped",
}

FAQ = {
    "shipping": "Standard shipping takes 5-7 business days.",
    "return": "You can return items within 30 days of delivery.",
    "refund": "Refunds are issued within 5-7 business days after we receive the return.",
    "payment": "We accept all major credit cards and PayPal.",
    "hours": "Our support team is available Mon-Fri, 9am-6pm.",
}


def lookup_order_status(text: str) -> str:
    for order_id in FAKE_ORDERS:
        if order_id in text:
            return f"Order {order_id}: {FAKE_ORDERS[order_id]}"
    return "I couldn't find an order ID in your message. Could you share your order number?"


def search_faq(text: str) -> str:
    text_lower = text.lower()
    for keyword, answer in FAQ.items():
        if keyword in text_lower:
            return answer
    return "I don't have a direct answer for that in our FAQ."


# ---------------------------------------------------------------------------
# 3. Intent classification — plain keyword rules (this replaces the LLM)
# ---------------------------------------------------------------------------
def classify_intent(state: SupportState) -> dict:
    last_msg = state["messages"][-1].content.lower()

    escalate_words = ["human", "agent", "manager", "angry", "furious", "terrible", "lawsuit"]
    order_words = ["order", "shipment", "tracking", "package", "delivery"]
    refund_words = ["refund", "money back", "reimburse"]
    faq_words = list(FAQ.keys())

    if any(w in last_msg for w in escalate_words):
        intent = "escalate"
    elif any(w in last_msg for w in order_words):
        intent = "order_status"
    elif any(w in last_msg for w in refund_words):
        intent = "refund"
    elif any(w in last_msg for w in faq_words):
        intent = "faq"
    else:
        intent = "other"

    return {"intent": intent}


# ---------------------------------------------------------------------------
# 4. Response nodes — one per intent, all rule-based
# ---------------------------------------------------------------------------
def handle_order_status(state: SupportState) -> dict:
    last_msg = state["messages"][-1].content
    reply = lookup_order_status(last_msg)
    return {"messages": [{"role": "assistant", "content": reply}]}


def handle_refund(state: SupportState) -> dict:
    reply = ("Refunds are typically processed within 5-7 business days once we "
             "receive your return. Want me to check a specific order's eligibility? "
             "Just give me the order number.")
    return {"messages": [{"role": "assistant", "content": reply}]}


def handle_faq(state: SupportState) -> dict:
    last_msg = state["messages"][-1].content
    reply = search_faq(last_msg)
    return {"messages": [{"role": "assistant", "content": reply}]}


def handle_other(state: SupportState) -> dict:
    reply = ("I'm not sure I understood that. I can help with order status, "
             "refunds, or general questions (shipping, returns, payment, hours). "
             "You can also type 'human' to talk to a person.")
    return {"messages": [{"role": "assistant", "content": reply}]}


def escalate_node(state: SupportState) -> dict:
    reply = ("I'm connecting you with a human support agent who can help further. "
             "Please hold on — someone will be with you shortly.")
    return {"messages": [{"role": "assistant", "content": reply}], "escalated": True}


# ---------------------------------------------------------------------------
# 5. Routing
# ---------------------------------------------------------------------------
def route_by_intent(state: SupportState) -> Literal["order_status", "refund", "faq", "escalate", "other"]:
    return state["intent"]


# ---------------------------------------------------------------------------
# 6. Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("order_status", handle_order_status)
    graph.add_node("refund", handle_refund)
    graph.add_node("faq", handle_faq)
    graph.add_node("other", handle_other)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent", route_by_intent,
        {
            "order_status": "order_status",
            "refund": "refund",
            "faq": "faq",
            "other": "other",
            "escalate": "escalate",
        },
    )

    for node in ["order_status", "refund", "faq", "other", "escalate"]:
        graph.add_edge(node, END)

    checkpointer = MemorySaver()  # keeps conversation history per thread_id
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 7. Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_graph()
    config = {"configurable": {"thread_id": "user-123"}}

    print("Support bot ready (rule-based, no API/AI). Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"quit", "exit"}:
            break
        result = app.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        print("Bot:", result["messages"][-1].content, "\n")

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import sqlite3

from policy_retrieval import retrieve

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "instamart.db"
LEDGER_PATH = ROOT / "outputs" / "refund_ledger.jsonl"


def get_order(order_id):
    """Look up an order in the cleaned SQLite database."""
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

    if row is None:
        return {"error": "order not found"}

    return dict(row)


def search_policy(query):
    """Retrieve the three policy chunks most relevant to a query."""
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Calculate the late-delivery refund for a completed order."""
    amount = float(order_amount)

    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Order amount must be finite and non-negative")

    # Status eligibility is checked by the agent before this tool.
    if delivery_minutes is None:
        raise ValueError("Delivery time is required for refund calculation")

    minutes = float(delivery_minutes)

    if not math.isfinite(minutes) or minutes < 0:
        raise ValueError("Delivery time must be finite and non-negative")

    refund = round(amount, 2) if minutes > 45 else 0.0
    return {"refund_inr": refund}


def validate_refund_action(order_id, refund_inr):
    """Check an action against the stored order before recording it."""
    order = get_order(order_id)

    if "error" in order:
        raise ValueError("Order not found")

    if order["status"] != "completed":
        raise ValueError("Only completed orders can be refunded")

    expected = calculate_refund(
        order["order_amount"],
        order["delivery_minutes"],
    )["refund_inr"]

    amount = float(refund_inr)

    if (
        not math.isfinite(amount)
        or expected <= 0
        or abs(amount - expected) > 0.000001
    ):
        raise ValueError("Refund must match the policy-calculated amount")

    return expected


def append_ledger(tool_name, order_id, refund_inr):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr,
    }

    with LEDGER_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


def issue_refund(order_id, refund_inr):
    """Record an eligible automatic refund of at most INR 1000."""
    amount = validate_refund_action(order_id, refund_inr)

    if amount > 1000:
        raise ValueError("Refunds above INR 1000 require human approval")

    append_ledger("issue_refund", order_id, amount)
    return {"refund_inr": amount}


def escalate_to_human(order_id, refund_inr):
    """Record an eligible refund above INR 1000 for human approval."""
    amount = validate_refund_action(order_id, refund_inr)

    if amount <= 1000:
        raise ValueError("This tool handles refunds above INR 1000")

    append_ledger("escalate_to_human", order_id, amount)
    return {"refund_inr": amount}


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Look up an order's status, amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve relevant policy passages for a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate a completed order's late-delivery refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Record an eligible refund of at most INR 1000.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund above INR 1000 for human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def main():
    assert len(TOOLS_SPEC) == 5
    assert {tool["name"] for tool in TOOLS_SPEC} == set(TOOL_MAP)

    assert calculate_refund(500, 55) == {"refund_inr": 500.0}
    assert calculate_refund(500, 45) == {"refund_inr": 0.0}
    assert calculate_refund(500, 30) == {"refund_inr": 0.0}

    assert get_order("ORD-000000") == {"error": "order not found"}

    result = search_policy("refund policy late delivery")
    assert len(result["results"]) == 3

    print("Tools defined: 5")
    print("Late delivery: INR 500 refund")
    print("Delivery at exactly 45 minutes: INR 0 refund")
    print("Delivery within 45 minutes: INR 0 refund")
    print("Unknown order lookup: passed")
    print("Policy retrieval: passed")
    print("No refund actions were recorded during these checks.")
    print("REFUND TOOL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
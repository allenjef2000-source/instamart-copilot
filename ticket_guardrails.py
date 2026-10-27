from pathlib import Path
import re

import pandas as pd
from mock_llm import est_tokens

ROOT = Path(__file__).resolve().parent
MAX_INPUT_TOKENS = 400

# Apply these patterns in the assignment's required order.
PATTERNS = [
    ("email", r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),
    ("phone", r"\b[6-9]\d{9}\b", "[PHONE]"),
    ("order_id", r"ORD-\d{6}", "[ORDER_ID]"),
]


def check_input(text):
    """Check the original ticket before any model call."""
    if not text.strip():
        return "skipped_empty"

    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"

    return "eligible"


def redact_text(text):
    """Mask personal information and count each replacement."""
    counts = {}

    for name, pattern, placeholder in PATTERNS:
        text, count = re.subn(pattern, placeholder, text)
        counts[name] = count

    # No matching personal information may remain.
    for _, pattern, _ in PATTERNS:
        assert re.search(pattern, text) is None, \
            "Personal information remains after redaction"

    return text, counts


def main():
    tickets = pd.read_csv(
        ROOT / "data" / "raw" / "tickets.csv",
        keep_default_na=False,
    )

    status_counts = {
        "skipped_empty": 0,
        "flagged_long": 0,
        "eligible": 0,
    }

    totals = {"email": 0, "phone": 0, "order_id": 0}
    remaining_matches = 0
    long_token_counts = []

    for ticket in tickets.itertuples(index=False):
        text = ticket.text

        # Check length before redaction.
        status = check_input(text)
        status_counts[status] += 1

        if status == "flagged_long":
            long_token_counts.append(est_tokens(text))

        # Audit redaction across all 164 tickets.
        redacted, counts = redact_text(text)

        for name in totals:
            totals[name] += counts[name]

        for _, pattern, _ in PATTERNS:
            remaining_matches += len(re.findall(pattern, redacted))

    assert len(tickets) == 164
    assert status_counts["skipped_empty"] == 4
    assert status_counts["flagged_long"] == 4
    assert status_counts["eligible"] == 156
    assert totals == {"email": 55, "phone": 91, "order_id": 132}
    assert sum(totals.values()) == 278
    assert remaining_matches == 0

    print("INPUT GUARDRAILS")
    print(f"Total tickets: {len(tickets)}")

    for status, count in status_counts.items():
        print(f"{status}: {count}")

    print(f"Long-ticket token estimates: {long_token_counts}")

    print("\nREDACTION SUMMARY")
    print(f"Emails replaced: {totals['email']}")
    print(f"Phone numbers replaced: {totals['phone']}")
    print(f"Order IDs replaced: {totals['order_id']}")
    print(f"Total replacements: {sum(totals.values())}")
    print(f"Personal-information matches remaining: {remaining_matches}")

    print("\nNo model calls were made.")
    print("TICKET GUARDRAIL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
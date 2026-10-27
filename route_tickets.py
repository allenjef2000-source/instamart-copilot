from pathlib import Path
import random

import pandas as pd
import mock_llm

from classify_tickets import call_with_retry
from ticket_guardrails import check_input, redact_text
from triage_prompts import (
    ZERO_SHOT_PROMPT,
    FEW_SHOT_PROMPT,
    validate_reply,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

REVIEW_STATUSES = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted",
}


def choose_route(status, category, is_premium):
    # Human-review rules take precedence over all other routing.
    if status in REVIEW_STATUSES:
        return "human_review", "NORMAL"

    if category == "fraud":
        return "security_team", "HIGH"

    if is_premium:
        return "premium_desk", "HIGH"

    return "support_queue", "NORMAL"


def run_triage(prompt, token_budget, filename):
    tickets = pd.read_csv(
        ROOT / "data" / "raw" / "tickets.csv",
        keep_default_na=False,
    )

    segments = pd.read_csv(
        OUTPUTS / "customer_segments.csv",
        keep_default_na=False,
    )

    assert segments["customer_id"].is_unique

    segment_lookup = segments.set_index("customer_id").to_dict("index")

    rows = []
    total_tokens = 0
    total_retries = 0
    rng = random.Random(42)

    mock_llm.reset()

    for ticket in tickets.itertuples(index=False):
        status = check_input(ticket.text)
        category = ""
        tokens_used = 0

        segment = segment_lookup.get(ticket.customer_id, {})
        segment_name = segment.get("segment_name", "")
        is_premium = (
            str(segment.get("is_premium", False)).strip().lower()
            == "true"
        )

        if status == "eligible":
            # Check the accumulated usage before making a call.
            if total_tokens >= token_budget:
                status = "budget_exhausted"

            else:
                redacted, _ = redact_text(ticket.text)

                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": redacted},
                ]

                response, retries = call_with_retry(
                    messages, ticket.ticket_id, rng
                )

                total_retries += retries
                tokens_used = response.usage.total_tokens
                total_tokens += tokens_used

                validated = validate_reply(
                    response.choices[0].message.content
                )

                status = validated["status"]
                category = validated["category"]

        route, priority = choose_route(
            status, category, is_premium
        )

        rows.append({
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": tokens_used,
        })

    results = pd.DataFrame(rows)

    assert len(results) == 164
    assert results["tokens_used"].sum() == total_tokens
    assert results["ticket_id"].tolist() == tickets["ticket_id"].tolist()

    no_call = results["status"].isin(
        ["skipped_empty", "flagged_long", "budget_exhausted"]
    )
    assert results.loc[no_call, "tokens_used"].eq(0).all()

    review = results["status"].isin(REVIEW_STATUSES)
    assert results.loc[review, "route"].eq("human_review").all()
    assert results.loc[review, "priority"].eq("NORMAL").all()

    # Independently check every row's routing against its segment.
    for row in results.itertuples(index=False):
        segment = segment_lookup.get(row.customer_id, {})
        premium = (
            str(segment.get("is_premium", False)).strip().lower()
            == "true"
        )

        expected_route, expected_priority = choose_route(
            row.status, row.category, premium
        )

        assert row.route == expected_route
        assert row.priority == expected_priority

    results.to_csv(OUTPUTS / filename, index=False)

    print(f"\nRUN: {filename}")
    print(f"Token budget: {token_budget}")
    print(f"Tokens used: {total_tokens}")
    print(f"Recovered temporary failures: {total_retries}")

    print("\nStatus counts:")
    print(results["status"].value_counts().to_string())

    print("\nRouting counts:")
    print(results.groupby(["route", "priority"]).size().to_string())

    print(f"\nSaved: outputs/{filename}")

    return results


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Demonstrate stopping further model calls after the budget is reached.
    demo = run_triage(
        ZERO_SHOT_PROMPT,
        token_budget=6000,
        filename="triage_budget_demo.csv",
    )

    exhausted = int(demo["status"].eq("budget_exhausted").sum())
    assert exhausted > 0, "Budget demo must exhaust at least one ticket"

    # Final deliverable uses few-shot classification and a larger budget.
    final = run_triage(
        FEW_SHOT_PROMPT,
        token_budget=100000,
        filename="triage_results.csv",
    )

    counts = final["status"].value_counts()

    assert counts.get("classified_json", 0) == 135
    assert counts.get("classified_keyword", 0) == 12
    assert counts.get("flagged_invalid_output", 0) == 9
    assert counts.get("budget_exhausted", 0) == 0

    print(f"\nBudget-demo tickets exhausted: {exhausted}")
    print("BUDGET AND ROUTING CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
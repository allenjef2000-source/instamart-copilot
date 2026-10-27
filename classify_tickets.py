from pathlib import Path
import random
import time

import pandas as pd
import mock_llm

from ticket_guardrails import check_input, redact_text
from triage_prompts import (
    ZERO_SHOT_PROMPT,
    FEW_SHOT_PROMPT,
    validate_reply,
)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
BASE_DELAY = 0.05
MAX_RETRIES = 3


def call_with_retry(messages, ticket_id, rng):
    """One initial attempt, followed by at most three retries."""
    retries = 0

    while True:
        try:
            response = mock_llm.chat(messages)
            return response, retries

        except mock_llm.TransientLLMError:
            if retries >= MAX_RETRIES:
                raise

            # Retry numbers are 0, 1 and 2.
            wait = BASE_DELAY * (2 ** retries) * rng.uniform(0.9, 1.15)

            print(
                f"RETRY ticket={ticket_id} "
                f"retry={retries} wait={wait:.6f}s"
            )

            time.sleep(wait)
            retries += 1


def run_classification(prompt, run_name):
    tickets = pd.read_csv(
        ROOT / "data" / "raw" / "tickets.csv",
        keep_default_na=False,
    )

    rng = random.Random(42)
    rows = []
    total_tokens = 0
    total_retries = 0

    # Restart the mock immediately before each complete run.
    mock_llm.reset()

    # Preserve the CSV's original ticket order.
    for ticket in tickets.itertuples(index=False):
        status = check_input(ticket.text)
        category = ""
        tokens_used = 0

        if status == "eligible":
            redacted, _ = redact_text(ticket.text)

            messages = [
                {"role": "system", "content": prompt},
                # The user message is exactly the redacted text.
                {"role": "user", "content": redacted},
            ]

            response, retries = call_with_retry(
                messages, ticket.ticket_id, rng
            )

            total_retries += retries
            tokens_used = response.usage.total_tokens
            total_tokens += tokens_used

            reply = response.choices[0].message.content
            validated = validate_reply(reply)

            status = validated["status"]
            category = validated["category"]

        rows.append({
            "ticket_id": ticket.ticket_id,
            "customer_id": ticket.customer_id,
            "status": status,
            "category": category,
            "true_category": ticket.true_category,
            "tokens_used": tokens_used,
        })

    results = pd.DataFrame(rows)

    classified = results.loc[
        results["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    correct = int(
        classified["category"].eq(classified["true_category"]).sum()
    )
    accuracy = correct / len(classified)

    counts = results["status"].value_counts()

    assert counts.get("skipped_empty", 0) == 4
    assert counts.get("flagged_long", 0) == 4
    assert counts.get("classified_json", 0) == 135
    assert counts.get("classified_keyword", 0) == 12
    assert counts.get("flagged_invalid_output", 0) == 9
    assert total_retries == 19
    assert len(classified) == 147

    print(f"\n{run_name.upper()} RESULTS")
    print(counts.to_string())
    print(f"\nRecovered temporary failures: {total_retries}")
    print(f"Classified tickets: {len(classified)}")
    print(f"Correct classifications: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens per classified ticket: {total_tokens / len(classified):.2f}")

    if run_name == "zero_shot":
        assert correct == 126

        table = pd.crosstab(
            classified["true_category"],
            classified["category"],
            rownames=["Actual"],
            colnames=["Predicted"],
        ).reindex(
            index=["billing", "delivery", "technical", "fraud"],
            columns=["billing", "delivery", "technical", "fraud"],
            fill_value=0,
        )

        print("\nZERO-SHOT CONFUSION TABLE")
        print(table.to_string())
        table.to_csv(OUTPUTS / "zero_shot_confusion.csv")

    else:
        assert correct == 147

    results.to_csv(
        OUTPUTS / f"triage_{run_name}_comparison.csv",
        index=False,
    )

    return {
        "prompt": run_name,
        "classified": len(classified),
        "correct": correct,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "tokens_per_classified": total_tokens / len(classified),
        "retries": total_retries,
    }


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    zero_shot = run_classification(ZERO_SHOT_PROMPT, "zero_shot")
    few_shot = run_classification(FEW_SHOT_PROMPT, "few_shot")

    comparison = pd.DataFrame([zero_shot, few_shot])
    comparison.to_csv(
        OUTPUTS / "prompt_comparison.csv",
        index=False,
    )

    print("\nPROMPT COMPARISON")
    print(comparison.round(4).to_string(index=False))
    print("\nCLASSIFICATION CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
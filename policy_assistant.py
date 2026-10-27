from pathlib import Path

import pandas as pd
import mock_llm

from policy_retrieval import PolicyRetriever

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

SYSTEM_PROMPT = """POLICY ASSISTANT
Answer only using the supplied policy context.
If the context does not contain the answer, say:
I could not find that information in the policy documents.
"""

_default_retriever = None


def answer(question, retriever=None):
    global _default_retriever

    if retriever is None:
        if _default_retriever is None:
            _default_retriever = PolicyRetriever(overlap=10)

        retriever = _default_retriever

    chunks = retriever.retrieve(question, k=3)

    context = "\n".join(
        f"[{chunk['source']}]\n{chunk['text']}"
        for chunk in chunks
    )

    user_message = (
        f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    )

    response = mock_llm.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])

    return {
        "text": response.choices[0].message.content,
        "sources": [chunk["source"] for chunk in chunks],
    }


def evaluate(overlap):
    questions = pd.read_csv(
        ROOT / "data" / "raw" / "policy_eval.csv",
        keep_default_na=False,
    )

    retriever = PolicyRetriever(overlap=overlap)
    rows = []

    mock_llm.reset()

    for row in questions.itertuples(index=False):
        result = answer(row.question, retriever=retriever)
        answerable = row.expected_doc != "NONE"

        rows.append({
            "question": row.question,
            "expected_doc": row.expected_doc,
            "expected_fact": row.expected_fact,
            "answerable": answerable,
            "retrieved_sources": " | ".join(result["sources"]),
            "retrieval_hit": (
                row.expected_doc in result["sources"]
                if answerable else False
            ),
            "answer": result["text"],
            "fact_hit": (
                row.expected_fact in result["text"]
                if answerable else False
            ),
            "fallback": result["text"].startswith(
                "I could not find"
            ),
        })

    results = pd.DataFrame(rows)

    covered = results.loc[results["answerable"]]
    uncovered = results.loc[~results["answerable"]]

    retrieval_hits = int(covered["retrieval_hit"].sum())
    fact_hits = int(covered["fact_hit"].sum())
    fallbacks = int(uncovered["fallback"].sum())

    print(f"\nEVALUATION — OVERLAP {overlap}")
    print(f"Total chunks: {len(retriever.chunks)}")
    print(f"Expected document retrieved: {retrieval_hits}/12")
    print(f"Expected fact in answer: {fact_hits}/12")
    print(f"Unanswerable questions given fallback: {fallbacks}/2")

    failures = covered.loc[
        ~covered["retrieval_hit"] | ~covered["fact_hit"]
    ]

    if not failures.empty:
        print("\nANSWERABLE QUESTIONS THAT FAILED")

        for row in failures.itertuples(index=False):
            print(f"\nQuestion: {row.question}")
            print(f"Expected fact: {row.expected_fact}")
            print(f"Answer: {row.answer}")
            print(f"Sources: {row.retrieved_sources}")

    results.to_csv(
        OUTPUTS / f"policy_eval_overlap_{overlap}.csv",
        index=False,
    )

    return {
        "overlap": overlap,
        "chunks": len(retriever.chunks),
        "retrieval_hits": retrieval_hits,
        "fact_hits": fact_hits,
        "unanswerable_fallbacks": fallbacks,
    }


def main():
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    overlapping = evaluate(overlap=10)
    no_overlap = evaluate(overlap=0)

    summary = pd.DataFrame([overlapping, no_overlap])

    summary.to_csv(
        OUTPUTS / "policy_evaluation_summary.csv",
        index=False,
    )

    print("\nPOLICY EVALUATION COMPARISON")
    print(summary.to_string(index=False))

    assert overlapping["retrieval_hits"] == 12
    assert overlapping["fact_hits"] == 12
    assert overlapping["unanswerable_fallbacks"] == 2

    assert (
        no_overlap["retrieval_hits"] < 12
        or no_overlap["fact_hits"] < 12
    ), "Expected at least one answerable question to fail without overlap"

    print("\nSaved detailed results in outputs/.")
    print("POLICY ASSISTANT CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
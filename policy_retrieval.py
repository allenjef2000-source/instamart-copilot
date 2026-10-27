from pathlib import Path
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
POLICY_DIR = ROOT / "data" / "policies"


def chunk_document(text, chunk_size=40, overlap=10):
    """Split text into word chunks, stopping at the first final chunk."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be between 0 and chunk_size - 1")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))

        if end >= len(words):
            break

    return chunks


class PolicyRetriever:
    def __init__(self, overlap=10):
        self.chunks = []

        for path in sorted(POLICY_DIR.glob("*.txt")):
            text = path.read_text(encoding="utf-8")

            for chunk in chunk_document(text, overlap=overlap):
                self.chunks.append({
                    "source": path.name,
                    "text": chunk,
                })

        if not self.chunks:
            raise ValueError("No policy text was found")

        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(
            query_vector, self.matrix
        ).ravel()

        # Use chunk order to break equal-score ties consistently.
        ranked = sorted(
            range(len(self.chunks)),
            key=lambda index: (-scores[index], index),
        )[:k]

        return [
            {
                "source": self.chunks[index]["source"],
                "score": float(scores[index]),
                "text": self.chunks[index]["text"],
            }
            for index in ranked
        ]


# Load the default retriever only when first needed.
_default_retriever = None


def retrieve(query, k=3):
    global _default_retriever

    if _default_retriever is None:
        _default_retriever = PolicyRetriever(overlap=10)

    return _default_retriever.retrieve(query, k=k)


def main():
    retriever = PolicyRetriever(overlap=10)

    counts = Counter(
        chunk["source"] for chunk in retriever.chunks
    )

    expected = {
        "refund_policy.txt": 4,
        "delivery_policy.txt": 3,
        "escalation_policy.txt": 3,
        "payment_policy.txt": 3,
        "privacy_policy.txt": 2,
    }

    print("POLICY CHUNK COUNTS")

    for source, count in sorted(counts.items()):
        print(f"{source}: {count}")

    print(f"\nTotal chunks: {len(retriever.chunks)}")

    assert dict(counts) == expected, "Policy chunk counts differ"
    assert len(retriever.chunks) == 15

    query = "refund policy late delivery"
    results = retriever.retrieve(query, k=3)

    print(f"\nSEARCH: {query}")

    for rank, result in enumerate(results, start=1):
        print(
            f"\nResult {rank}: {result['source']} "
            f"| score={result['score']:.4f}"
        )
        print(result["text"])

    assert len(results) == 3
    assert all(
        result["source"] == "refund_policy.txt"
        for result in results
    )

    print("\nPOLICY RETRIEVAL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
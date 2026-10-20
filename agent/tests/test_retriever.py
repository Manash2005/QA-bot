from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.retrieval.retriever import DocumentRetriever


def main():
    retriever = DocumentRetriever()

    query = "What percentage of a child's cumulative brain development occurs before the age of 6?"

    results = retriever.retrieve(
        query=query,
        top_k=3,
    )

    if not results:
        print("No documents found. Index some documents first.")
        return

    for i, result in enumerate(results, start=1):
        print(f"\n--- Result {i} ---")
        print("Text:", result["text"][:300])
        print("Metadata:", result["metadata"])
        print("Cosine distance:", result["distance"])


if __name__ == "__main__":
    main()
from functools import lru_cache
from langchain_core.tools import tool
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.retrieval.retriever import DocumentRetriever


@lru_cache(maxsize=1)
def get_retriever() -> DocumentRetriever:
    return DocumentRetriever()


@tool
def search_knowledge_base(query: str, top_k: int = 5) -> str:
    """Search uploaded PDF documents for relevant facts to answer a question."""
    if not query.strip():
        return "Query cannot be empty."

    retriever = get_retriever()
    results = retriever.retrieve(query=query, top_k=min(max(top_k, 1), 8))

    if not results:
        return "No relevant documents found."

    formatted = []
    for r in results:
        source = r["metadata"].get("source", "Unknown")
        page = r["metadata"].get("page", 0)
        page_num = page + 1 if isinstance(page, int) else page

        formatted.append(f"[{source} - Page {page_num}]\n{r['text']}")

    return "\n\n---\n\n".join(formatted)
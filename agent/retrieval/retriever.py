from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "documents"


class DocumentRetriever:
    def __init__(self):
        # 1. Attach the same embedding function used during indexing
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            normalize_embeddings=True,
        )

        # 2. Connect to local ChromaDB collection
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []

        total_docs = self.collection.count()
        if total_docs == 0:
            return []

        # Chroma automatically embeds `query_texts` behind the scenes
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, total_docs),
        )

        # Flatten Chroma's nested list output into clean dictionaries
        return [
            {
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i, doc in enumerate(results["documents"][0])
        ]
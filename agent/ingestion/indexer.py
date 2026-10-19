from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from langchain_core.documents import Document

CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "documents"


class DocumentIndexer:
    def __init__(self):
        # 1. Setup embedding model
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            normalize_embeddings=True,
        )

        # 2. Connect to ChromaDB
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_documents(self, documents: list[Document]) -> int:
        # Ignore empty documents
        docs = [d for d in documents if d.page_content.strip()]
        if not docs:
            return 0

        # SIMPLEST ID: Combine source file name with chunk index number
        # Example: "resume.pdf_chunk_0", "resume.pdf_chunk_1"
        ids = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "doc")
            ids.append(f"{source}_chunk_{i}")

        # Extract text content
        texts = [doc.page_content for doc in docs]

        # Keep only basic metadata values (Chroma requirement)
        metadatas = [
            {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
            for doc in docs
        ]

        # Save to Chroma
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        return len(docs)

    def count(self) -> int:
        return self.collection.count()
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.ingestion.indexer import DocumentIndexer
from agent.ingestion.chunker import chunk_documents
from agent.ingestion.loader import load_pdf

FILE_PATH = ROOT / "data" / "raw" / "NEP_2020.pdf"

def main():
    documents = load_pdf(FILE_PATH)
    print(f"Pages loaded: {len(documents)}")

    chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=50)
    print(f"chunks created : {len(chunks)}")

    indexer = DocumentIndexer()

    indexed_count = indexer.index_documents(chunks)

    print(f"Chunks processed : {indexed_count}")
    print(f"Total chunks in databse : {indexer.count()}")


if __name__ == "__main__":
    main()


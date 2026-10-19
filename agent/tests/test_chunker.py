from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.ingestion.chunker import chunk_documents
from agent.ingestion.loader import load_pdf

FILE_PATH = ROOT / "data" / "raw" / "NEP_2020.pdf"

documents = load_pdf(FILE_PATH)

chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=50)

print(f"Pages loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")



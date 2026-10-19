from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.ingestion.loader import load_pdf

FILE_PATH = ROOT / "data" / "raw" / "NEP_2020.pdf"

documents = load_pdf(FILE_PATH)

for document in documents:
    print(document.page_content)
    print(document.metadata)
    print("--------------------------------------------------")
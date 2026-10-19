from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(documents : list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ["\n\n", "\n", ". ", " ", ""],
        length_function = len,
        add_start_index = True
    )

    chunks = text_splitter.split_documents(documents)

    return chunks
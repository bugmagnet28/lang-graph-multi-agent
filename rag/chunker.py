from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"[Chunker] Split into {len(chunks)} chunks")
    return chunks

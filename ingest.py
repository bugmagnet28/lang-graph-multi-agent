from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.vectorstore import build_vectorstore

if __name__ == "__main__":
    print(" Loading documents from data/...")
    docs = load_documents("data/")

    print("\n Chunking documents...")
    chunks = chunk_documents(docs)

    print("\nEmbedding and saving to vectorstore...")
    build_vectorstore(chunks)

    print("\n Ingestion complete! Run `python main.py` to start chatting.")

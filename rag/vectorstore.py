import os
from langchain_chroma import Chroma
from core.embeddings import get_embeddings

PERSIST_DIR = "vectorstore/chroma_db"

def build_vectorstore(chunks):
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print(f"[VectorStore] Built and saved to {PERSIST_DIR}")
    return vectorstore

def load_vectorstore():
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(
            "No vectorstore found. Run `python main.py --ingest` first."
        )
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    print(f"[VectorStore] Loaded from {PERSIST_DIR}")
    return vectorstore

def get_retriever(vectorstore, k: int = 4):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

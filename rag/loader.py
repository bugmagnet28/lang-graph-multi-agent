import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents(data_dir: str = "data/"):
    documents = []

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue

        docs = loader.load()
        documents.extend(docs)
        print(f"[Loader] Loaded: {filename} ({len(docs)} pages/sections)")

    print(f"[Loader] Total documents loaded: {len(documents)}")
    return documents

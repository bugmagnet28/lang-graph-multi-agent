from langchain_core.tools import tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from core.llm import get_llm

def build_rag_tool(retriever):

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a document assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found in documents."

Context:
{context}

Question: {input}
""")

    combine_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_chain)

    @tool
    def search_documents(query: str) -> str:
        """Search through the user's uploaded documents to answer questions.
        Use this when the question is about personal documents, uploaded files,
        internal knowledge, or anything that might be in the user's documents.
        For general internet facts, use the web search tool instead.
        """
        result = rag_chain.invoke({"input": query})
        answer = result["answer"]
        sources = result["context"]

        source_info = []
        for doc in sources:
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "document")
            source_info.append(f"Page {page} of {source}")

        sources_text = " | ".join(set(source_info))
        return f"{answer}\n\nSources: {sources_text}"

    return search_documents

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever
from core.llm import get_llm

session_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

def build_rag_chain(retriever):
    llm = get_llm()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the chat history and the latest user question,
        rewrite it as a standalone question that makes sense without the history.
        Do NOT answer it — just rephrase if needed, otherwise return as-is."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer the question using ONLY
        the context below. If the answer isn't in the context, say:
        "I don't have enough information in the provided documents to answer that."

        Context:
        {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return conversational_chain

def ask(chain, question: str, session_id: str = "default"):
    config = {"configurable": {"session_id": session_id}}
    result = chain.invoke({"input": question}, config=config)
    return result["answer"], result["context"]

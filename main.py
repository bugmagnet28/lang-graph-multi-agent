from langchain_core.messages import HumanMessage
from rag.vectorstore import load_vectorstore, get_retriever
from tools import get_all_tools
from agents.graph import build_graph
from memory import new_session

def main():
    print("🔄 Loading vectorstore...")
    try:
        vectorstore = load_vectorstore()
        retriever = get_retriever(vectorstore, k=4)
        print("✅ Documents loaded — agent can search your files")
    except FileNotFoundError:
        print("⚠️  No documents found. Run `python ingest.py` to add documents.")
        print("   Continuing with web search and calculator only.\n")
        retriever = None

    tools = get_all_tools(retriever) if retriever else get_all_tools_no_rag()
    graph = build_graph(tools, with_memory=True)
    session_id = new_session()
    config = {"configurable": {"thread_id": session_id}}

    print("\n" + "="*60)
    print("🤖 ReAct + RAG Agent")
    print(f"   Session: {session_id}")
    print("   Tools: web search | document search | calculator")
    print("   Type 'exit' to quit")
    print("="*60)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        for chunk in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values"
        ):
            last = chunk["messages"][-1]
            if hasattr(last, "content") and last.content and not getattr(last, "tool_calls", None):
                print(f"\nAgent: {last.content}")
                break

def get_all_tools_no_rag():
    from tools.search_tool import search
    from tools.calculator_tool import calculator
    return [search, calculator]

if __name__ == "__main__":
    main()

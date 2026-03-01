from langchain_core.messages import HumanMessage
from agents.graph import build_graph
from memory import new_session, DEFAULT_SESSION

def run_single_query(query: str):
    """Run a one-shot query with no memory."""
    graph = build_graph(with_memory=False)
    result = graph.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content


def run_conversation(session_id: str = DEFAULT_SESSION):
    """Run an interactive multi-turn conversation with memory."""
    graph = build_graph(with_memory=True)
    config = {"configurable": {"thread_id": session_id}}

    print("=" * 60)
    print("🤖 ReAct Agent — Multi-Turn Conversation Mode")
    print(f"   Session ID: {session_id}")
    print("   Type 'exit' to quit | 'new' to start a new session")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        if user_input.lower() == "new":
            session_id = new_session()
            config = {"configurable": {"thread_id": session_id}}
            print(f"New session started: {session_id}")
            continue

        print("\nAgent: ", end="", flush=True)

        # Stream response token by token
        for chunk in graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="values"
        ):
            last = chunk["messages"][-1]
            # Only print final LLM text responses, not tool call internals
            if hasattr(last, "content") and last.content and not getattr(last, "tool_calls", None):
                print(f"\nAgent: {last.content}")
                break


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Single query mode: python main.py "What is the GDP of India?"
        query = " ".join(sys.argv[1:])
        print(f"\nQuery: {query}")
        print(f"\nAnswer: {run_single_query(query)}")
    else:
        # Interactive conversation mode
        run_conversation(session_id=new_session())

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from core.state import AgentState
from agents.nodes import call_model, tool_node, should_continue

def build_graph(with_memory: bool = True):
    builder = StateGraph(AgentState)

    builder.add_node("llm", call_model)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "llm")

    builder.add_conditional_edges(
        "llm",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )

    builder.add_edge("tools", "llm")

    if with_memory:
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)

    return builder.compile()

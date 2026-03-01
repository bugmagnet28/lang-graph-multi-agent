from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from core.llm import get_llm
from core.state import AgentState
from tools import all_tools
from langgraph.graph import END

llm = get_llm()
llm_with_tools = llm.bind_tools(all_tools)

SYSTEM_PROMPT = """You are a helpful research assistant with access to web search and a calculator.

Always follow this reasoning process:
1. Think about what information you need
2. Use search to find real-time facts
3. Use calculator for any math operations
4. Only give a final answer when you are fully confident

Be concise and accurate in your final answer."""

def call_model(state: AgentState) -> dict:
    system = SystemMessage(content=SYSTEM_PROMPT)
    response = llm_with_tools.invoke([system] + state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(all_tools)

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

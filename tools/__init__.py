from tools.search_tool import search
from tools.calculator_tool import calculator
from tools.rag_tool import build_rag_tool

def get_all_tools(retriever):
    rag_tool = build_rag_tool(retriever)
    return [search, calculator, rag_tool]

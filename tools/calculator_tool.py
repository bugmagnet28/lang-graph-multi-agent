from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression.
    Input must be a valid Python math expression like '1400000000 * 0.125'.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"

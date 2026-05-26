from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict):
    input: str
    output: str

def node(state: State):
    return {
        "output": f"echo: {state['input']}"
    }

graph = StateGraph(State)

graph.add_node("node", node)

graph.set_entry_point("node")
graph.add_edge("node", END)

app_graph = graph.compile()

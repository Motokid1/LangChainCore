from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END

class LoopState(TypedDict):
    count: int
    logs: Annotated[list, operator.add]

def loop_node(state: LoopState):
    return {
        "count": state["count"] + 1,
        "logs": [f"step {state['count']}"]
    }

def condition(state: LoopState) -> Literal["loop", "end"]:
    if state["count"] < 3:
        return "loop"
    return "end"

graph = StateGraph(LoopState)

graph.add_node("loop", loop_node)

graph.add_edge(START, "loop")

graph.add_conditional_edges(
    "loop",
    condition,
    {
        "loop": "loop",
        "end": END
    }
)

app = graph.compile()

print(app.invoke({"count": 0, "logs": []}))
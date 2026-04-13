from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    input: str

def router(state: State) -> Literal["positive", "negative"]:
    if "good" in state["input"]:
        return "positive"
    return "negative"

def positive_node(state: State):
    return {"input": "😊 Positive path"}

def negative_node(state: State):
    return {"input": "😞 Negative path"}

graph = StateGraph(State)

graph.add_node("positive", positive_node)
graph.add_node("negative", negative_node)

graph.add_conditional_edges(
    START,
    router,
    {
        "positive": "positive",
        "negative": "negative"
    }
)

graph.add_edge("positive", END)
graph.add_edge("negative", END)

app = graph.compile()

print(app.invoke({"input": "this is good"}))
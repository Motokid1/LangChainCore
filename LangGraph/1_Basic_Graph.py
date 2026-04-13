from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class MyState(TypedDict):
    text: str

def step1(state: MyState):
    return {"text": state["text"] + " -> step1"}

def step2(state: MyState):
    return {"text": state["text"] + " -> step2"}

graph = StateGraph(MyState)

graph.add_node("step1", step1)
graph.add_node("step2", step2)

graph.add_edge(START, "step1")
graph.add_edge("step1", "step2")
graph.add_edge("step2", END)

app = graph.compile()

result = app.invoke({"text": "start"})
print(result)
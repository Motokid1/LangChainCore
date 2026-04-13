from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    text: str

def step1(state: State):
    return {"text": state["text"] + " -> step1"}

def human_step(state: State):
    print("Current state:", state)
    input("Approve? Press Enter to continue...")
    return state

def step2(state: State):
    return {"text": state["text"] + " -> step2"}

graph = StateGraph(State)

graph.add_node("step1", step1)
graph.add_node("human", human_step)
graph.add_node("step2", step2)

graph.add_edge(START, "step1")
graph.add_edge("step1", "human")
graph.add_edge("human", "step2")
graph.add_edge("step2", END)

app = graph.compile()

print(app.invoke({"text": "start"}))
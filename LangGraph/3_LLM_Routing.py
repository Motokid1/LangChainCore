from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

class Route(BaseModel):
    route: Literal["math", "general"]

class State(TypedDict):
    input: str

structured_llm = llm.with_structured_output(Route)

def router(state: State) -> Literal["math", "general"]:
    decision = structured_llm.invoke(state["input"])
    return decision.route

def math_node(state: State):
    return {"input": "Math route selected"}

def general_node(state: State):
    return {"input": "General route selected"}

graph = StateGraph(State)

graph.add_node("math", math_node)
graph.add_node("general", general_node)

graph.add_conditional_edges(
    START,
    router,
    {
        "math": "math",
        "general": "general"
    }
)

graph.add_edge("math", END)
graph.add_edge("general", END)

app = graph.compile()

print(app.invoke({"input": "what is 2+2?"}))
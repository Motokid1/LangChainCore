from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, Literal
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    step: int
    logs: Annotated[list, operator.add]

def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "logs": ["chatbot called"]
    }

def loop_node(state: State):
    return {
        "step": state["step"] + 1,
        "logs": [f"loop {state['step']}"]
    }

def router(state: State) -> Literal["loop", "end"]:
    if state["step"] < 2:
        return "loop"
    return "end"

graph = StateGraph(State)

graph.add_node("chatbot", chatbot)
graph.add_node("loop", loop_node)

graph.add_edge(START, "chatbot")

graph.add_conditional_edges(
    "chatbot",
    router,
    {
        "loop": "loop",
        "end": END
    }
)

graph.add_edge("loop", "chatbot")

app = graph.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}

result = app.invoke(
    {"messages": [("user", "Tell me about RAG in 1 line")], "step": 0, "logs": []},
    config=config
)

print(f"\nFinal AI Response: {result['messages'][-1].content}")
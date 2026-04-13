from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

class ChatState(TypedDict):
    messages: Annotated[list, add_messages]

memory = InMemorySaver()

def chatbot(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "user1"}}

app.invoke({"messages": [("user", "Hi")]}, config=config)
result = app.invoke({"messages": [("user", "How are you?")]}, config=config)

print(result)
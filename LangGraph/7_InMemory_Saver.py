from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END, MessageState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")

def chatbot(state: MessageState):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(MessageState)

graph.add_node("chatbot", chatbot)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "1"}}

print(app.invoke({"messages": [("user", "Hello")]}, config=config))
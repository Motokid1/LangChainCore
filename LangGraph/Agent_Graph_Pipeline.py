# User Query
#    ↓
# Smart Router (LLM decides)
#    ↓
#  ┌───────────┬───────────┬────────────┐
#  │ simple    │ rag       │ live       │
#  ↓           ↓           ↓
# LLM       RAG tool     Weather tool
#  ↓           ↓           ↓
#  END       Agent       Agent
from typing import TypedDict, Annotated, Literal
import operator
import requests
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

# RAG Imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# LOAD ENV
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# STATE
class GraphState(TypedDict):
    query: str
    location: str
    weather: str
    rag_result: str
    decision: Literal["rag", "live"]
    final_answer: str
    messages: Annotated[list, operator.add]

# TOOLS

@tool
def get_location() -> str:
    """Fetch the current user location based on IP address."""
    res = requests.get("http://ip-api.com/json/", timeout=5).json()
    return f"{res.get('city')}, {res.get('country')}"

@tool
def get_weather(location: str) -> str:
    """Fetch current weather details for a given location."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    data = requests.get(url, timeout=5).json()

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]

    return f"Weather in {location}: {temp}°C, {desc}"

# 🔥 IMPROVED RAG TOOL
@tool
def rag_pipeline(query: str) -> str:
    """
    Extract location from documents using RAG and fetch its weather.
    """

    try:
        folder = "D:\\Gen AI\\data"

        # Load documents
        loader = DirectoryLoader(
            folder,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        docs = loader.load()

        # Add PDFs
        for file in os.listdir(folder):
            if file.endswith(".pdf"):
                pdf_loader = PyPDFLoader(os.path.join(folder, file))
                docs.extend(pdf_loader.load())

        if not docs:
            return "No documents found"

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        split_docs = splitter.split_documents(docs)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector DB
        vectordb = Chroma.from_documents(
            split_docs,
            embeddings,
            persist_directory="./chroma_db"
        )

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        results = retriever.invoke(query)

        if not results:
            return "No relevant documents found"

        context = "\n\n".join([doc.page_content for doc in results])

        # 🔥 STEP 1: Extract location using LLM
        extraction_prompt = f"""
        Extract the location name from the context below.

        Context:
        {context}

        Return ONLY the location name. If none found, return 'NONE'.
        """

        location_response = llm.invoke(extraction_prompt)
        location = location_response.content.strip()

        if location == "NONE":
            return "No location found in documents"

        # 🔥 STEP 2: Call weather API
        weather = get_weather.invoke({"location": location})

        return f"📍 Location found in documents: {location}\n{weather}"

    except Exception as e:
        return f"RAG error: {str(e)}"

# LLM (Groq)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b"
)

agent = create_react_agent(
    llm,
    tools=[get_location, get_weather, rag_pipeline]
)

# NODES

def router_node(state: GraphState):
    return {"next": "human_decision"}

#Older version with input prompt
# def human_node(state: GraphState):
#     print("\nUser Query:", state["query"])
#     choice = input("Choose data source (rag/live): ").strip().lower()

#     if choice not in ["rag", "live"]:
#         print("Invalid choice. Defaulting to 'rag'")
#         choice = "rag"

    # return {"decision": choice}

def smart_router_node(state: GraphState):
    query = state["query"]

    prompt = f"""
    Classify the user query into one of these:

    1. 'simple' → general questions (answer directly)
    2. 'rag' → needs document search
    3. 'live' → needs real-time data like weather/location

    Query: {query}

    Answer ONLY one word: simple / rag / live
    """

    decision = llm.invoke(prompt).content.strip().lower()

    if decision not in ["simple", "rag", "live"]:
        decision = "simple"

    return {"decision": decision}

def simple_node(state: GraphState):
    response = llm.invoke(state["query"])

    return {
        "final_answer": response.content,
        "messages": ["Simple LLM response"]
    }


def decision_router(state: GraphState):
    return state["decision"]


def live_node(state: GraphState):
    location = get_location.invoke({})
    weather = get_weather.invoke({"location": location})

    return {
        "location": location,
        "weather": weather,
        "messages": ["Live data fetched"]
    }


def rag_node(state: GraphState):
    result = rag_pipeline.invoke({"query": state["query"]})
    return {
        "rag_result": result,
        "messages": ["RAG executed"]
    }


def agent_node(state: GraphState):
    response = agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })

    return {
        "final_answer": response["messages"][-1].content,
        "messages": ["Agent executed"]
    }


def final_node(state: GraphState):
    answer = ""

    if state.get("weather"):
        answer += state["weather"] + "\n"
    if state.get("rag_result"):
        answer += state["rag_result"] + "\n"
    if state.get("final_answer"):
        answer += state["final_answer"]

    return {"final_answer": answer}

# GRAPH
builder = StateGraph(GraphState)

builder.add_node("router", router_node)
builder.add_node("smart_router", smart_router_node)
builder.add_node("live", live_node)
builder.add_node("rag", rag_node)
builder.add_node("agent", agent_node)
builder.add_node("final", final_node)
builder.add_node("simple", simple_node)
builder.add_edge("simple", END)

builder.add_edge(START, "router")
builder.add_edge(START, "smart_router")
builder.add_conditional_edges(
    "smart_router",
    decision_router,
    {
        "simple": "simple",
        "rag": "rag",
        "live": "live"
    }
)

builder.add_edge("live", "agent")
builder.add_edge("rag", "agent")

builder.add_edge("agent", "final")
builder.add_edge("final", END)

# MEMORY
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# RUN
if __name__ == "__main__":
    while True:
        q = input("\nEnter Query: ")

        result = graph.invoke(
            {
                "query": q,
                "messages": []
            },
            config={
                "configurable": {
                    "thread_id": "user_1"
                }
            }
        )

        print("\nFinal Answer:\n", result["final_answer"])

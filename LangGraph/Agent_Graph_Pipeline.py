"""
Agentic Weather + RAG Pipeline using LangGraph
================================================
Flow:
    User Query
        ↓
    smart_router  ← LLM classifies query into: simple | rag | live
        ↓
   ┌────┬────┬──────┐
   │    │    │      │
simple  rag  live   │
   │    │    │      │
  END  agent_node ←─┘
         ↓
        final
         ↓
        END

Conversation Memory Design
--------------------------
MemorySaver (LangGraph checkpointer) saves the graph's *execution state*
between runs — useful for resuming interrupted workflows — but it does NOT
automatically inject past Q&A turns into the LLM's prompt.

To give the LLM actual memory of "what the user said before", we maintain a
`chat_history` list of {"role": ..., "content": ...} dicts OUTSIDE the graph
and pass the full list into every invoke() call via GraphState.
The nodes then forward that history to the LLM so it sees the whole dialogue.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import operator
from typing import TypedDict, Annotated, Literal

# ── Third-party ───────────────────────────────────────────────────────────────
import requests
from dotenv import load_dotenv

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver   # ← Fix: use MemorySaver, not InMemorySaver
from langgraph.prebuilt import create_react_agent

# ── LangChain ─────────────────────────────────────────────────────────────────
from langchain.tools import tool
from langchain_groq import ChatGroq

# ── RAG dependencies ──────────────────────────────────────────────────────────
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ══════════════════════════════════════════════════════════════════════════════
# 0. ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
DOCS_FOLDER         = os.getenv("DOCS_FOLDER", r"D:\Gen AI\data")  # override via .env
CHROMA_DIR          = "./chroma_db"


# ══════════════════════════════════════════════════════════════════════════════
# 1. GRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    """
    Shared state that flows through every node of the graph.

    Fields
    ------
    query        : The raw user question for this turn.
    chat_history : Full conversation so far as LangChain message dicts
                   [{"role": "user"|"assistant", "content": "..."}].
                   Passed into every LLM call so the model remembers
                   previous turns (e.g. the user's name).
    location     : City/country resolved by IP or RAG.
    weather      : Human-readable weather string.
    rag_result   : Output from the RAG pipeline tool.
    decision     : Router decision — 'simple' | 'rag' | 'live'.
    final_answer : The answer shown to the user at the end.
    messages     : Append-only log of node activity (for debugging).
    """
    query:        str
    chat_history: list                            # list of {"role", "content"} dicts
    location:     str
    weather:      str
    rag_result:   str
    decision:     Literal["simple", "rag", "live"]
    final_answer: str
    messages:     Annotated[list, operator.add]   # operator.add = append semantics


# ══════════════════════════════════════════════════════════════════════════════
# 2. LLM
# ══════════════════════════════════════════════════════════════════════════════

# Initialise once so every node/tool shares the same client instance.
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_location() -> str:
    """
    Detect the user's approximate location from their public IP address.
    Returns a 'City, Country' string (e.g. 'Hyderabad, India').
    """
    response = requests.get("http://ip-api.com/json/", timeout=5).json()
    city    = response.get("city", "Unknown")
    country = response.get("country", "Unknown")
    return f"{city}, {country}"


@tool
def get_weather(location: str) -> str:
    """
    Fetch current weather for *location* via OpenWeatherMap.
    Returns a short summary string with temperature (°C) and description.
    """
    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        f"?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    data = requests.get(url, timeout=5).json()

    # Safely pull out the fields we need
    temp = data.get("main", {}).get("temp", "N/A")
    desc = data.get("weather", [{}])[0].get("description", "N/A")

    return f"Weather in {location}: {temp}°C, {desc}"


@tool
def rag_pipeline(query: str) -> str:
    """
    Retrieve documents from the local knowledge base, extract a location
    mentioned in them, then fetch live weather for that location.

    Steps
    -----
    1. Load .txt and .pdf files from DOCS_FOLDER.
    2. Split + embed → Chroma vector store.
    3. Semantic search for *query*.
    4. Use the LLM to extract a place name from the retrieved chunks.
    5. Call get_weather for that place.
    """
    try:
        # ── 3a. Load documents ────────────────────────────────────────────────
        txt_loader = DirectoryLoader(
            DOCS_FOLDER,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
        )
        docs = txt_loader.load()

        # Also pick up any PDFs in the same folder
        for filename in os.listdir(DOCS_FOLDER):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(DOCS_FOLDER, filename)
                docs.extend(PyPDFLoader(pdf_path).load())

        if not docs:
            return "⚠️ No documents found in the configured folder."

        # ── 3b. Chunk documents ───────────────────────────────────────────────
        splitter   = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)

        # ── 3c. Embed + store in Chroma ───────────────────────────────────────
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectordb = Chroma.from_documents(
            split_docs,
            embeddings,
            persist_directory=CHROMA_DIR,
        )

        # ── 3d. Semantic retrieval ────────────────────────────────────────────
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        results   = retriever.invoke(query)

        if not results:
            return "⚠️ No relevant chunks found for the query."

        context = "\n\n".join(doc.page_content for doc in results)

        # ── 3e. Extract location with LLM ─────────────────────────────────────
        extraction_prompt = f"""
Extract the location name from the context below.

Context:
{context}

Return ONLY the location name. If no location is found, return the word NONE.
"""
        location = llm.invoke(extraction_prompt).content.strip()

        if location.upper() == "NONE":
            return "⚠️ No location found in the retrieved documents."

        # ── 3f. Fetch weather for the extracted location ───────────────────────
        weather = get_weather.invoke({"location": location})
        return f"📍 Location found in documents: **{location}**\n{weather}"

    except Exception as exc:
        return f"❌ RAG error: {exc}"


# ══════════════════════════════════════════════════════════════════════════════
# 4. REACT AGENT  (wraps all three tools; used for rag + live branches)
# ══════════════════════════════════════════════════════════════════════════════

agent = create_react_agent(
    llm,
    tools=[get_location, get_weather, rag_pipeline],
)


# ══════════════════════════════════════════════════════════════════════════════
# 5. GRAPH NODES
# ══════════════════════════════════════════════════════════════════════════════

def smart_router_node(state: GraphState) -> dict:
    """
    Classify the user query and decide which branch to take:
      - 'simple' → answer directly with the LLM (no tools)
      - 'rag'    → search local documents, extract location, fetch weather
      - 'live'   → detect IP location in real-time and fetch weather

    Chat history is included so the router can classify follow-up questions
    correctly (e.g. "what about tomorrow?" after a weather query → 'live').
    """
    # Build a readable summary of recent history for the classifier
    history_text = ""
    if state.get("chat_history"):
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in state["chat_history"][-6:]   # last 3 turns is enough context
        )
        history_text = f"\nConversation so far:\n{history_text}\n"

    classification_prompt = f"""{history_text}
Classify the latest user query into exactly one category:

  simple → general knowledge / conversational question (no tools needed)
  rag    → answer requires searching local documents
  live   → answer requires real-time data like current weather or location

Latest query: {state["query"]}

Reply with a SINGLE lowercase word: simple / rag / live
"""
    raw_decision = llm.invoke(classification_prompt).content.strip().lower()

    # Guard against unexpected model output
    decision = raw_decision if raw_decision in ("simple", "rag", "live") else "simple"
    return {"decision": decision}


def decision_router(state: GraphState) -> str:
    """
    Conditional edge function — returns the next node name
    based on the decision stored in state.
    """
    return state["decision"]


def simple_node(state: GraphState) -> dict:
    """
    Handle 'simple' queries: pass the FULL chat history + current query to
    the LLM so it can reference anything the user said earlier in the session.
    No tools are invoked.
    """
    # Reconstruct the full dialogue as a message list for the LLM.
    # chat_history already contains all previous turns; append the new query.
    messages = list(state.get("chat_history", []))
    messages.append({"role": "user", "content": state["query"]})

    response = llm.invoke(messages)
    return {
        "final_answer": response.content,
        "messages":     ["[simple_node] Answered with full chat history"],
    }


def live_node(state: GraphState) -> dict:
    """
    Handle 'live' queries:
      1. Detect the user's location via IP.
      2. Fetch current weather for that location.
    Results are stored in state so agent_node / final_node can use them.
    """
    location = get_location.invoke({})
    weather  = get_weather.invoke({"location": location})
    return {
        "location": location,
        "weather":  weather,
        "messages": [f"[live_node] Fetched live weather for {location}"],
    }


def rag_node(state: GraphState) -> dict:
    """
    Handle 'rag' queries:
      Run the full RAG pipeline (load docs → embed → retrieve → extract location → weather).
    """
    result = rag_pipeline.invoke({"query": state["query"]})
    return {
        "rag_result": result,
        "messages":   ["[rag_node] RAG pipeline executed"],
    }


def agent_node(state: GraphState) -> dict:
    """
    Post-processing agent that runs after live_node or rag_node.
    The ReAct agent can call any tool if additional reasoning is needed
    before the final answer is assembled.
    """
    response = agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {
        "final_answer": response["messages"][-1].content,
        "messages":     ["[agent_node] ReAct agent completed"],
    }


def final_node(state: GraphState) -> dict:
    """
    Assemble the final answer from whichever fields were populated
    (weather, rag_result, final_answer). Concatenates them in order.
    """
    parts = []

    if state.get("weather"):
        parts.append(state["weather"])

    if state.get("rag_result"):
        parts.append(state["rag_result"])

    if state.get("final_answer"):
        parts.append(state["final_answer"])

    return {"final_answer": "\n".join(parts)}


# ══════════════════════════════════════════════════════════════════════════════
# 6. GRAPH DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

builder = StateGraph(GraphState)

# ── Register nodes ────────────────────────────────────────────────────────────
builder.add_node("smart_router", smart_router_node)
builder.add_node("simple",       simple_node)
builder.add_node("live",         live_node)
builder.add_node("rag",          rag_node)
builder.add_node("agent",        agent_node)
builder.add_node("final",        final_node)

# ── Entry point ───────────────────────────────────────────────────────────────
builder.add_edge(START, "smart_router")

# ── Routing: smart_router → one of {simple, rag, live} ───────────────────────
builder.add_conditional_edges(
    "smart_router",
    decision_router,
    {
        "simple": "simple",
        "rag":    "rag",
        "live":   "live",
    },
)

# ── simple branch ends immediately ────────────────────────────────────────────
builder.add_edge("simple", END)

# ── rag / live both pass through the ReAct agent before final ─────────────────
builder.add_edge("live",  "agent")
builder.add_edge("rag",   "agent")
builder.add_edge("agent", "final")
builder.add_edge("final", END)


# ══════════════════════════════════════════════════════════════════════════════
# 7. COMPILE WITH MEMORY
# ══════════════════════════════════════════════════════════════════════════════
#
# Fix: LangGraph ≥ 0.2 renamed InMemorySaver → MemorySaver.
# Always import from langgraph.checkpoint.memory.
# The checkpointer must be passed to .compile(), NOT to graph.invoke().
# Each unique thread_id in the config gets its own isolated memory.
# ──────────────────────────────────────────────────────────────────────────────

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════════════════════
# 8. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🤖 Agentic Weather + RAG Pipeline — type 'quit' to exit\n")

    # ── Conversation memory ───────────────────────────────────────────────────
    # This list grows with every turn and is passed into graph.invoke() each
    # time so every node (especially simple_node) sees the full dialogue.
    #
    # Why not rely solely on MemorySaver?
    # MemorySaver checkpoints the graph's *execution state* (nodes visited,
    # partial results) so a run can be resumed after a crash. It does NOT
    # replay the LLM prompt with prior Q&A turns — that's our job here.
    # ─────────────────────────────────────────────────────────────────────────
    chat_history: list = []

    # One thread_id per session keeps MemorySaver checkpoints isolated.
    RUN_CONFIG = {"configurable": {"thread_id": "user_1"}}

    while True:
        query = input("Enter Query: ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not query:
            continue

        result = graph.invoke(
            {
                "query":        query,
                "chat_history": chat_history,   # ← full history every time
                "messages":     [],
            },
            config=RUN_CONFIG,
        )

        answer = result["final_answer"]

        # Append this turn to history so the next invoke sees it
        chat_history.append({"role": "user",      "content": query})
        chat_history.append({"role": "assistant", "content": answer})

        print("\n── Final Answer ─────────────────────────────────────")
        print(answer)
        print("─────────────────────────────────────────────────────\n")
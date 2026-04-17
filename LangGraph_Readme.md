# Concepts Explained — Agentic Weather Pipeline

A thorough breakdown of every concept, library, and design decision used in the code — excluding RAG-related topics.

---

## Table of Contents

1. [LangGraph](#1-langgraph)
   - [StateGraph](#11-stategraph)
   - [GraphState & TypedDict](#12-graphstate--typeddict)
   - [Nodes](#13-nodes)
   - [Edges](#14-edges)
   - [Conditional Edges](#15-conditional-edges)
   - [START and END](#16-start-and-end)
   - [compile()](#17-compile)
2. [MemorySaver (Checkpointing)](#2-memorysaver-checkpointing)
   - [What MemorySaver Actually Does](#21-what-memorysaver-actually-does)
   - [Why It Does NOT Give the LLM Memory](#22-why-it-does-not-give-the-llm-memory)
   - [The Real Fix — chat_history](#23-the-real-fix--chat_history)
3. [Python Typing Concepts](#3-python-typing-concepts)
   - [TypedDict](#31-typeddict)
   - [Annotated](#32-annotated)
   - [Literal](#33-literal)
   - [operator.add as a Reducer](#34-operatoradd-as-a-reducer)
4. [LangChain Tools (@tool decorator)](#4-langchain-tools-tool-decorator)
   - [What @tool Does](#41-what-tool-does)
   - [tool.invoke()](#42-toolinvoke)
5. [ReAct Agent (create_react_agent)](#5-react-agent-create_react_agent)
   - [What ReAct Means](#51-what-react-means)
   - [How create_react_agent Works](#52-how-create_react_agent-works)
   - [Agent vs Node — When to Use Which](#53-agent-vs-node--when-to-use-which)
6. [LLM — ChatGroq](#6-llm--chatgroq)
   - [What Groq Is](#61-what-groq-is)
   - [llm.invoke()](#62-llminvoke)
   - [Passing Message Lists vs Plain Strings](#63-passing-message-lists-vs-plain-strings)
7. [Smart Router — LLM-Based Classification](#7-smart-router--llm-based-classification)
8. [Environment Variables & dotenv](#8-environment-variables--dotenv)
9. [External API Calls with requests](#9-external-api-calls-with-requests)
   - [ip-api (IP Geolocation)](#91-ip-api-ip-geolocation)
   - [OpenWeatherMap API](#92-openweathermap-api)
10. [Conversation Memory Design](#10-conversation-memory-design)
11. [Graph Execution Flow (End-to-End)](#11-graph-execution-flow-end-to-end)
12. [thread_id and Multi-User Isolation](#12-thread_id-and-multi-user-isolation)

---

## 1. LangGraph

**LangGraph** is a library built on top of LangChain that lets you define AI workflows as a **directed graph** — a set of nodes (processing steps) connected by edges (transitions). It is designed for building stateful, multi-step agentic applications.

Think of it like a flowchart where each box is a Python function and the arrows between boxes are controlled either statically (always go here next) or dynamically (decide at runtime where to go next).

---

### 1.1 StateGraph

```python
from langgraph.graph import StateGraph
builder = StateGraph(GraphState)
```

`StateGraph` is the main class you use to build a graph. You pass it your **state schema** (the `TypedDict` class) so it knows which fields are available throughout the workflow.

Every node in the graph receives the current state as input and returns a dictionary of fields to update in that state. LangGraph merges the returned dict into the existing state automatically.

---

### 1.2 GraphState & TypedDict

```python
class GraphState(TypedDict):
    query:        str
    chat_history: list
    decision:     Literal["simple", "rag", "live"]
    final_answer: str
    messages:     Annotated[list, operator.add]
    ...
```

`GraphState` is the **single source of truth** that travels through the entire graph. Instead of passing individual arguments between functions, every node reads from and writes back to this shared state object.

Using `TypedDict` (from Python's `typing` module) enforces that the state is a plain dictionary with specific typed keys — it is not a class with methods. This makes it serialisable and easy to checkpoint.

---

### 1.3 Nodes

```python
def simple_node(state: GraphState) -> dict:
    response = llm.invoke(messages)
    return {"final_answer": response.content}
```

A **node** is just a Python function that:

- Takes the full `GraphState` as its only argument
- Does some work (calls an LLM, runs a tool, fetches an API)
- Returns a **partial dict** of only the fields it wants to update

LangGraph merges the returned dict into the current state. Fields not returned are left unchanged.

Nodes are registered with:

```python
builder.add_node("simple", simple_node)
```

---

### 1.4 Edges

```python
builder.add_edge("agent", "final")
builder.add_edge("final", END)
```

An **edge** is a fixed connection between two nodes. When node A finishes, control always passes to node B. Use `add_edge()` when the next step is always the same regardless of what happened in the current node.

---

### 1.5 Conditional Edges

```python
builder.add_conditional_edges(
    "smart_router",
    decision_router,        # ← a function that inspects state and returns a string
    {
        "simple": "simple",
        "rag":    "rag",
        "live":   "live",
    },
)
```

A **conditional edge** lets the graph branch at runtime. Instead of a fixed next-node, you provide:

1. A **routing function** (`decision_router`) that looks at the current state and returns a string key
2. A **mapping dict** from that string key to the actual next node name

This is how the smart router sends different queries to different branches.

```python
def decision_router(state: GraphState) -> str:
    return state["decision"]   # returns "simple", "rag", or "live"
```

---

### 1.6 START and END

```python
from langgraph.graph import START, END

builder.add_edge(START, "smart_router")
builder.add_edge("simple", END)
```

`START` and `END` are special sentinel nodes provided by LangGraph:

- **`START`** — the virtual entry point. Connect it to your first real node.
- **`END`** — the virtual exit point. Connect a node to `END` when it is the last step in its branch.

A graph can have multiple paths to `END` (e.g., the `simple` branch ends immediately, while `rag` and `live` both pass through `agent → final → END`).

---

### 1.7 compile()

```python
graph = builder.compile(checkpointer=checkpointer)
```

`compile()` validates the graph (checks for disconnected nodes, missing edges, etc.) and returns a runnable object. The optional `checkpointer` argument wires in persistent state saving between invocations (see Section 2).

After compiling, you run the graph with:

```python
result = graph.invoke(initial_state, config=RUN_CONFIG)
```

---

## 2. MemorySaver (Checkpointing)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

> **Note:** In LangGraph ≥ 0.2 the class was renamed from `InMemorySaver` to `MemorySaver`. Always import from `langgraph.checkpoint.memory`.

---

### 2.1 What MemorySaver Actually Does

`MemorySaver` is a **graph execution checkpointer**. After each node completes, it serialises the current `GraphState` and stores it in memory keyed by `thread_id`. This serves two purposes:

1. **Fault tolerance** — if a long-running graph crashes mid-way, it can be resumed from the last checkpoint rather than restarting from scratch.
2. **Pause-and-resume** — in human-in-the-loop workflows, the graph can pause waiting for human input and then resume later.

---

### 2.2 Why It Does NOT Give the LLM Memory

This is the most common misconception. `MemorySaver` saves the **graph's execution state** (which nodes ran, what fields were set). It does **not** replay past Q&A turns into the LLM's prompt.

When you call `graph.invoke({"query": "what is my name?", ...})`, the LLM inside `simple_node` only sees what you put in the `messages` argument — it has no awareness of the previous invocation unless you explicitly pass that history in.

---

### 2.3 The Real Fix — chat_history

```python
# Outside the graph, in the REPL loop:
chat_history: list = []

# After each turn:
chat_history.append({"role": "user",      "content": query})
chat_history.append({"role": "assistant", "content": answer})

# Passed into every graph.invoke():
graph.invoke({"query": query, "chat_history": chat_history, ...})
```

```python
# Inside simple_node:
messages = list(state.get("chat_history", []))
messages.append({"role": "user", "content": state["query"]})
response = llm.invoke(messages)
```

By maintaining `chat_history` as a list outside the graph and threading it through `GraphState`, every LLM call receives the full conversation — giving it genuine memory of what the user said in previous turns.

---

## 3. Python Typing Concepts

### 3.1 TypedDict

```python
from typing import TypedDict

class GraphState(TypedDict):
    query: str
    decision: str
```

`TypedDict` creates a dictionary type with specific named keys and their expected types. Unlike a regular class, instances are just plain dicts at runtime — no methods, no `__init__`. This makes them lightweight and easy to serialise (important for checkpointing).

Static type checkers (like `mypy` or Pylance) use the annotations to catch typos and wrong field types at development time.

---

### 3.2 Annotated

```python
from typing import Annotated
messages: Annotated[list, operator.add]
```

`Annotated[T, metadata]` attaches extra metadata to a type hint. The first argument is the actual type (`list`), and anything after it is extra information that tools can use.

LangGraph specifically looks for `Annotated[list, operator.add]` on state fields and treats those fields as **append-only** — instead of overwriting the list, it calls `operator.add` (list concatenation) to merge the new value with the existing one. This prevents nodes from accidentally clobbering each other's log entries.

---

### 3.3 Literal

```python
from typing import Literal
decision: Literal["simple", "rag", "live"]
```

`Literal` restricts a field to only a specific set of allowed values. If you try to set `decision = "unknown"`, a type checker will flag it as an error. It serves as self-documenting code and catches bugs early.

---

### 3.4 operator.add as a Reducer

```python
import operator
messages: Annotated[list, operator.add]
```

`operator.add` is a function reference to Python's `+` operator. For lists, `list + list` concatenates them. LangGraph uses this as a **reducer** — a function that specifies how to merge a new value into an existing state field instead of replacing it.

Without a reducer, if two nodes both update `messages`, the second write overwrites the first. With `operator.add`, both writes are preserved in order.

---

## 4. LangChain Tools (@tool decorator)

### 4.1 What @tool Does

```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Fetch current weather for a given location."""
    ...
```

The `@tool` decorator converts a plain Python function into a **LangChain Tool object** — a structured wrapper that includes:

- The function's name (used by the LLM to decide when to call it)
- The docstring (used as the tool's description — the LLM reads this to understand what the tool does)
- The input schema (auto-generated from the function signature)

This is what makes tools usable by a ReAct agent: the agent can read all registered tool descriptions and decide which one to call based on the user's query.

---

### 4.2 tool.invoke()

```python
location = get_location.invoke({})
weather  = get_weather.invoke({"location": location})
```

When calling a tool directly (not through an agent), you use `.invoke()` and pass arguments as a dictionary. This is different from calling the raw function because `.invoke()` validates inputs and runs any LangChain middleware (like tracing or error handling).

---

## 5. ReAct Agent (create_react_agent)

### 5.1 What ReAct Means

**ReAct** stands for **Reason + Act**. It is a prompting pattern where the LLM alternates between:

1. **Thought** — reasoning about what to do next ("I need to find the user's location before fetching weather")
2. **Action** — calling a tool with specific arguments
3. **Observation** — reading the tool's output
4. **Repeat** — until the LLM decides it has enough information to give a final answer

This loop allows the LLM to solve multi-step problems that require external data.

---

### 5.2 How create_react_agent Works

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    llm,
    tools=[get_location, get_weather, rag_pipeline],
)
```

`create_react_agent` builds a pre-configured LangGraph subgraph that implements the ReAct loop. Internally it:

1. Sends the user message + all tool descriptions to the LLM
2. If the LLM requests a tool call, executes that tool
3. Feeds the result back to the LLM as an "observation"
4. Repeats until the LLM produces a plain text response (no more tool calls)

You invoke the agent with:

```python
response = agent.invoke({
    "messages": [{"role": "user", "content": state["query"]}]
})
final_text = response["messages"][-1].content
```

---

### 5.3 Agent vs Node — When to Use Which

| Use a plain **Node**                 | Use an **Agent**                                    |
| ------------------------------------ | --------------------------------------------------- |
| Fixed, deterministic logic           | Dynamic tool selection based on context             |
| You always know exactly what to call | The LLM needs to decide which tools (if any) to use |
| Simple data transformation           | Multi-step reasoning with multiple tool calls       |

In this codebase, `live_node` and `rag_node` are plain nodes (they always do one fixed thing), while `agent_node` uses the ReAct agent for any additional reasoning needed after the initial data fetch.

---

## 6. LLM — ChatGroq

### 6.1 What Groq Is

**Groq** is an AI inference provider that runs large language models at very high speed using custom hardware (LPUs — Language Processing Units). `ChatGroq` is LangChain's integration class for the Groq API.

```python
from langchain_groq import ChatGroq

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="openai/gpt-oss-120b",
)
```

The LLM is instantiated once and shared across all nodes and tools, avoiding repeated connection overhead.

---

### 6.2 llm.invoke()

```python
response = llm.invoke("What is the capital of France?")
print(response.content)   # → "Paris"
```

`llm.invoke()` sends a prompt to the model and returns an `AIMessage` object. The actual text is in `.content`.

---

### 6.3 Passing Message Lists vs Plain Strings

```python
# Plain string — no conversation context:
llm.invoke("What is my name?")

# Message list — LLM sees the full conversation:
llm.invoke([
    {"role": "user",      "content": "My name is Rohith"},
    {"role": "assistant", "content": "Nice to meet you, Rohith!"},
    {"role": "user",      "content": "What is my name?"},
])
```

When you pass a **list of message dicts**, the LLM sees all prior turns and can reference them. Each dict has a `role` (`"user"` or `"assistant"`) and `"content"`. This is the standard OpenAI-style chat format that most LLM providers support.

Passing a plain string is equivalent to sending a single user message with no history — the LLM has zero context from previous turns.

---

## 7. Smart Router — LLM-Based Classification

```python
def smart_router_node(state: GraphState) -> dict:
    classification_prompt = f"""
Classify the latest user query into exactly one category:
  simple → general knowledge / conversational question
  rag    → answer requires searching local documents
  live   → answer requires real-time data like weather or location

Latest query: {state["query"]}
Reply with a SINGLE lowercase word: simple / rag / live
"""
    raw_decision = llm.invoke(classification_prompt).content.strip().lower()
    decision = raw_decision if raw_decision in ("simple", "rag", "live") else "simple"
    return {"decision": decision}
```

Instead of writing `if "weather" in query` style keyword rules, the router uses the LLM itself as a classifier. The LLM reads the query, understands the intent, and returns a routing label.

**Why this is better than keyword matching:**

- Handles paraphrasing ("what's it like outside?" → `live`, no "weather" keyword)
- Understands follow-up context ("what about tomorrow?" after a weather query)
- No maintenance as query patterns evolve

**The fallback guard** (`if raw_decision not in (...)`) prevents unexpected model output from crashing the conditional edge.

---

## 8. Environment Variables & dotenv

```python
from dotenv import load_dotenv
import os

load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
```

**Why environment variables?**
API keys must never be hardcoded in source files — if the file is committed to git, the key is exposed publicly and can be abused.

**How dotenv works:**
`load_dotenv()` reads a `.env` file in the project root (e.g. `GROQ_API_KEY=gsk_abc123`) and injects those key-value pairs into the process's environment. `os.getenv()` then reads them at runtime.

Your `.env` file should be listed in `.gitignore` so it is never committed.

---

## 9. External API Calls with requests

### 9.1 ip-api (IP Geolocation)

```python
import requests

response = requests.get("http://ip-api.com/json/", timeout=5).json()
city    = response.get("city", "Unknown")
country = response.get("country", "Unknown")
```

`ip-api.com` is a free service that returns geolocation data based on the caller's public IP address. No API key is required. The `timeout=5` argument ensures the request fails fast (raises an exception) if the server doesn't respond within 5 seconds — important so the chatbot doesn't hang indefinitely.

`.json()` parses the raw HTTP response body as JSON and returns a Python dict. `.get("city", "Unknown")` safely reads the key with a fallback default if it's missing.

---

### 9.2 OpenWeatherMap API

```python
url = (
    "http://api.openweathermap.org/data/2.5/weather"
    f"?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
)
data = requests.get(url, timeout=5).json()

temp = data.get("main", {}).get("temp", "N/A")
desc = data.get("weather", [{}])[0].get("description", "N/A")
```

OpenWeatherMap's `/weather` endpoint returns current weather for a city name. Key query parameters:

| Parameter      | Purpose                                     |
| -------------- | ------------------------------------------- |
| `q`            | City name (e.g. `Hyderabad`)                |
| `appid`        | Your API key                                |
| `units=metric` | Returns temperature in °C instead of Kelvin |

**Defensive chaining** with `.get()` prevents `KeyError` crashes if the API returns an error response (e.g. city not found, invalid key):

- `data.get("main", {})` returns an empty dict if `"main"` is missing
- `.get("temp", "N/A")` then returns `"N/A"` instead of crashing

---

## 10. Conversation Memory Design

This is the most architecturally important concept in the codebase.

```
Turn 1:  User: "My name is Rohith"        → chat_history = []
         Bot:  "Nice to meet you, Rohith!"
         → append both to chat_history

Turn 2:  User: "What is my name?"         → chat_history = [turn 1]
         LLM receives: [user: "My name is Rohith", assistant: "Nice to meet you...", user: "What is my name?"]
         Bot:  "Your name is Rohith."
```

The key design decision is that `chat_history` lives **outside the graph**, in the `__main__` REPL loop. This is intentional:

- The graph is designed to handle **one query at a time**
- `MemorySaver` preserves the graph's internal execution state but does not manage dialogue history
- By maintaining history externally and injecting it into `GraphState` on every `graph.invoke()`, all nodes that call the LLM can receive full context

---

## 11. Graph Execution Flow (End-to-End)

Here is what happens when you type `"What's the weather like?"`:

```
graph.invoke({"query": "What's the weather like?", "chat_history": [...], ...})
        │
        ▼
[START] ──► [smart_router_node]
                │  LLM classifies query → "live"
                │  state["decision"] = "live"
                ▼
        decision_router() returns "live"
                │
                ▼
        [live_node]
                │  get_location.invoke({}) → "Hyderabad, India"
                │  get_weather.invoke({"location": "Hyderabad, India"}) → "28°C, clear sky"
                │  state["location"] = "Hyderabad, India"
                │  state["weather"] = "Weather in Hyderabad: 28°C, clear sky"
                ▼
        [agent_node]
                │  ReAct agent processes the query with tools available
                │  state["final_answer"] = agent's response text
                ▼
        [final_node]
                │  Combines state["weather"] + state["final_answer"]
                │  state["final_answer"] = assembled answer
                ▼
        [END]  →  result["final_answer"] printed to user
```

---

## 12. thread_id and Multi-User Isolation

```python
RUN_CONFIG = {"configurable": {"thread_id": "user_1"}}

result = graph.invoke(initial_state, config=RUN_CONFIG)
```

`thread_id` is the key under which `MemorySaver` stores checkpoints. Each unique `thread_id` gets its own isolated execution history. This means:

- `"user_1"` and `"user_2"` running simultaneously will not see each other's state
- If the same `thread_id` resumes after a crash, it picks up from the last saved checkpoint
- In a production app, `thread_id` would typically be the user's session ID or database user ID

In the current single-user REPL, `"user_1"` is hardcoded since there is only ever one concurrent user.

---

_End of document_

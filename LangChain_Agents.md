# LangChain & LangGraph — Complete Reference Guide

A comprehensive reference covering LangChain's core concepts, agent patterns, and how everything connects to LangGraph.

---

## Table of Contents

1. [LLM Initialization — `init_chat_model()`](#1-llm-initialization--init_chat_model)
2. [Message System](#2-message-system)
3. [Tools](#3-tools)
4. [Agent Creation — `create_react_agent()`](#4-agent-creation--create_react_agent)
5. [Tool Calling Mechanism](#5-tool-calling-mechanism)
6. [Manual Agent Loop](#6-manual-agent-loop)
7. [Dynamic Tool Calling](#7-dynamic-tool-calling)
8. [ReAct Pattern](#8-react-pattern)
9. [Structured Output — `.with_structured_output()`](#9-structured-output--with_structured_output)
10. [Prompt-Based Agent Setup](#10-prompt-based-agent-setup)
11. [Chains — LCEL (LangChain Expression Language)](#11-chains--lcel-langchain-expression-language)
12. [Memory & Conversation History](#12-memory--conversation-history)
13. [Callbacks & Tracing](#13-callbacks--tracing)
14. [How Everything Connects to LangGraph](#14-how-everything-connects-to-langgraph)

---

## 1. LLM Initialization — `init_chat_model()`

This is your entry point. Instead of importing a specific model class (like `ChatOpenAI` or `ChatAnthropic`), `init_chat_model()` gives you a **provider-agnostic** way to create an LLM.

```python
from langchain.chat_models import init_chat_model

# Basic usage
llm = init_chat_model("gpt-4o", model_provider="openai")
llm = init_chat_model("claude-3-5-sonnet-20241022", model_provider="anthropic")
llm = init_chat_model("gemini-1.5-pro", model_provider="google_vertexai")
```

**Parameters:**

| Parameter | Description |
|---|---|
| `model` | Model name string (e.g. `"gpt-4o"`, `"claude-3-5-sonnet-..."`) |
| `model_provider` | `"openai"`, `"anthropic"`, `"google_vertexai"`, etc. |
| `temperature` | `0.0` (deterministic) to `1.0` (creative) |
| `max_tokens` | Max output tokens |
| `configurable_fields` | Which fields can be swapped at runtime |

The real power: you can swap providers **without changing your agent code downstream**.

---

## 2. Message System

LangChain models communicate through a typed **message list**. Every call to the LLM is just passing a list of messages. There are 4 core message types:

### `SystemMessage`
Sets the AI's persona and instructions. Placed first in the list. The LLM never "forgets" this context.

```python
from langchain_core.messages import SystemMessage

msg = SystemMessage(content="You are a helpful data analyst. Always return numbers as integers.")
```

### `HumanMessage`
Represents what the user said.

```python
from langchain_core.messages import HumanMessage

msg = HumanMessage(content="What is the capital of France?")
```

### `AIMessage`
Represents what the model replied. Crucially, it can also carry **tool call requests**.

```python
from langchain_core.messages import AIMessage

# Simple reply
msg = AIMessage(content="The capital of France is Paris.")

# Tool call reply (LLM asking to call a tool)
msg = AIMessage(
    content="",
    tool_calls=[{
        "name": "search_web",
        "args": {"query": "capital of France"},
        "id": "call_abc123"
    }]
)
```

### `ToolMessage`
The **result** of executing a tool. It must reference the `tool_call_id` from the `AIMessage` that requested it.

```python
from langchain_core.messages import ToolMessage

msg = ToolMessage(
    content="Paris",
    tool_call_id="call_abc123"   # must match the AIMessage's tool call id
)
```

### Message Flow in Practice

```python
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What's 2 + 2?"),
]

response = llm.invoke(messages)         # returns AIMessage
messages.append(response)               # add AI reply to history

messages.append(HumanMessage(content="Now multiply that by 3."))
response2 = llm.invoke(messages)        # LLM sees full conversation history
```

> **Key insight:** The message list **is** your memory. There's no magic state — you manually manage it.

---

## 3. Tools

Tools are **functions the LLM can decide to call**. The LLM doesn't run them — it returns a structured request saying "call this tool with these args." Your code then executes the function.

### `@tool` decorator — simplest approach

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers together."""   # docstring becomes the tool description
    return a * b

print(multiply.name)          # "multiply"
print(multiply.description)   # "Multiply two integers together."
print(multiply.args_schema)   # JSON schema generated from type hints
```

The `@tool` decorator:
- Reads **type hints** (`a: int, b: int`) → builds JSON schema automatically
- Reads the **docstring** → becomes the tool description the LLM sees
- Returns a `StructuredTool` object with `.invoke()`, `.name`, `.description`, `.args_schema`

### `StructuredTool` — more control

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="The search query to run")
    max_results: int = Field(default=5, description="Maximum number of results")

def search_web(query: str, max_results: int = 5) -> str:
    """Search the internet for information."""
    return f"Results for: {query}"

search_tool = StructuredTool.from_function(
    func=search_web,
    name="search_web",
    description="Search the internet for current information",
    args_schema=SearchInput,
)
```

Use `StructuredTool` over `@tool` when you need rich field descriptions via `Field(description=...)`, or when your function already exists and you don't want to modify it.

### `BaseTool` — full custom control

```python
from langchain_core.tools import BaseTool

class DatabaseQueryTool(BaseTool):
    name: str = "database_query"
    description: str = "Query the internal database for customer data"
    db_connection: str = "postgresql://..."  # custom attributes

    def _run(self, query: str) -> str:
        return f"DB result for: {query}"

    async def _arun(self, query: str) -> str:
        return f"Async DB result for: {query}"

db_tool = DatabaseQueryTool(db_connection="postgresql://mydb")
```

Use `BaseTool` when your tool needs **custom state**, async support, or complex initialization.

### `.invoke()` method

All tools share `.invoke()` regardless of how they were created:

```python
result = multiply.invoke({"a": 3, "b": 4})  # → 12

# Build a tool lookup map (used in manual loops)
tools = [multiply, search_tool, db_tool]
tool_map = {t.name: t for t in tools}
```

---

## 4. Agent Creation — `create_react_agent()`

This wires everything together: LLM + tools + system prompt → a runnable agent.

```python
from langgraph.prebuilt import create_react_agent

llm = init_chat_model("gpt-4o", model_provider="openai")
tools = [multiply, search_tool]

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier="You are a helpful assistant with access to tools."
)

# Run the agent
result = agent.invoke({
    "messages": [HumanMessage(content="What is 123 * 456?")]
})
print(result["messages"][-1].content)
```

**Parameters of `create_react_agent()`:**

| Parameter | Description |
|---|---|
| `model` | Your LLM (already initialized) |
| `tools` | List of tool objects |
| `state_modifier` | System prompt string, `SystemMessage`, or a state-modifying function |
| `checkpointer` | For memory/persistence (LangGraph feature) |
| `interrupt_before` / `interrupt_after` | For human-in-the-loop (LangGraph feature) |

> **Note:** `create_react_agent` from `langgraph.prebuilt` is preferred over the older `langchain.agents` version for production use.

---

## 5. Tool Calling Mechanism

When the LLM sees tools available (via `.bind_tools()`), it can return an `AIMessage` with `tool_calls` instead of a text reply.

### How `AIMessage.tool_calls` works

```python
llm_with_tools = llm.bind_tools([multiply, search_tool])

response = llm_with_tools.invoke([
    HumanMessage(content="What is 6 times 7?")
])

print(response.tool_calls)
# [{'name': 'multiply', 'args': {'a': 6, 'b': 7}, 'id': 'call_xyz789'}]

print(response.content)
# "" (empty — LLM chose to call a tool instead of answering directly)
```

### Tool Call Detection

```python
if response.tool_calls:
    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]      # which tool
        tool_args = tool_call["args"]      # what arguments
        tool_id   = tool_call["id"]        # unique call identifier

        result = tool_map[tool_name].invoke(tool_args)

        tool_msg = ToolMessage(
            content=str(result),
            tool_call_id=tool_id
        )
else:
    # LLM answered directly — no tools needed
    print(response.content)
```

---

## 6. Manual Agent Loop

`create_react_agent` handles looping automatically, but you can build the loop yourself for full control over what happens at each step.

```python
from langchain_core.messages import HumanMessage, ToolMessage

messages = [HumanMessage(content="Search for today's weather in London, then multiply the temperature by 2")]
tool_map = {t.name: t for t in tools}

while True:
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        print("Final answer:", response.content)
        break

    for tool_call in response.tool_calls:
        tool_fn = tool_map[tool_call["name"]]
        result  = tool_fn.invoke(tool_call["args"])

        messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    # Loop continues — LLM sees tool results and decides what to do next
```

This is the **exact loop** that `create_react_agent` runs internally. Building it manually lets you:
- Add custom logic between steps (logging, human approval, etc.)
- Modify tool results before feeding back to LLM
- Debug step-by-step

---

## 7. Dynamic Tool Calling

When you pass multiple tools, the LLM **automatically decides** which tool(s) to call, in what order, with what arguments — based purely on the user's request.

```python
tools = [
    search_web_tool,
    multiply_tool,
    get_weather_tool,
    lookup_database_tool,
]

agent = create_react_agent(model=llm, tools=tools)

# The LLM picks which tool(s) to use — you don't specify
result = agent.invoke({
    "messages": [HumanMessage(content="What's 15% tip on a $47.50 meal?")]
})
# LLM decides: use multiply_tool, not search_web_tool
```

The model's **function calling** capability (built into GPT-4, Claude, Gemini, etc.) handles this. Tools are passed as a JSON schema to the model API, and the model outputs which function to call.

---

## 8. ReAct Pattern

ReAct = **Reasoning + Acting**. The model interleaves thinking with tool use in a loop:

```
Thought: I need to find the weather in London first
Action: get_weather(city="London")
Observation: 18°C, partly cloudy

Thought: Now I need to multiply 18 by 2
Action: multiply(a=18, b=2)
Observation: 36

Thought: I have the answer
Final Answer: The temperature doubled is 36°C
```

In modern LLMs (GPT-4, Claude 3+), this reasoning is **implicit**. You can make it explicit with a prompt:

```python
from langchain_core.prompts import ChatPromptTemplate

react_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. When solving problems:
1. Think step by step before acting
2. Use tools when you need external information
3. Observe results before taking the next action
4. Only give a final answer when you're confident"""),
    ("placeholder", "{messages}")
])

agent = create_react_agent(
    model=llm,
    tools=tools,
    state_modifier=react_prompt
)
```

`create_react_agent` internally supports ReAct-style execution — each loop iteration is one **Thought → Action → Observation** cycle.

---

## 9. Structured Output — `.with_structured_output()`

Forces the LLM to return **typed, validated JSON** matching a Pydantic schema. No more parsing strings.

```python
from pydantic import BaseModel, Field
from typing import List

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating from 0 to 10")
    pros: List[str] = Field(description="List of positives")
    cons: List[str] = Field(description="List of negatives")
    summary: str = Field(description="One sentence summary")

structured_llm = llm.with_structured_output(MovieReview)

result = structured_llm.invoke("Review the movie Inception")

print(result.title)    # "Inception"   ← typed, not a string you parse
print(result.rating)   # 9.2           ← float, not "9.2/10"
print(result.pros)     # ["Brilliant plot", "Visual effects"]
```

**Parameters of `.with_structured_output()`:**

| Parameter | Description |
|---|---|
| `schema` | A Pydantic `BaseModel` class, or a JSON schema dict |
| `method` | `"function_calling"` (default) or `"json_mode"` |
| `include_raw` | If `True`, also returns the raw `AIMessage` alongside the parsed object |

---

## 10. Prompt-Based Agent Setup

Beyond `create_react_agent`, you can manually bind a system prompt and tools:

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial analyst. Today's date is {date}."),
    MessagesPlaceholder(variable_name="messages"),       # conversation history
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # tool results
])

# Chain: prompt → llm_with_tools
chain = prompt | llm.bind_tools(tools)

response = chain.invoke({
    "date": "2026-04-09",
    "messages": [HumanMessage(content="What's Apple's stock price?")],
    "agent_scratchpad": []
})
```

`MessagesPlaceholder` is a slot that gets filled with a list of messages at invocation time — it's how you inject conversation history and tool results into a prompt template.

---

## 11. Chains — LCEL (LangChain Expression Language)

LCEL uses the `|` pipe operator to compose components into **chains**. Every component that implements `Runnable` can be piped together.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

# Chain: prompt → llm → output parser
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"input": "Tell me a joke about Python"})
print(result)  # plain string output
```

### Common Output Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,       # returns plain string
    JsonOutputParser,      # parses JSON string → dict
    PydanticOutputParser,  # parses JSON → Pydantic model
)

# JSON output
json_chain = prompt | llm | JsonOutputParser()
result = json_chain.invoke({"input": "Return a JSON with name and age"})
print(result["name"])  # dict access

# Pydantic output
class Person(BaseModel):
    name: str
    age: int

pydantic_chain = prompt | llm | PydanticOutputParser(pydantic_object=Person)
result = pydantic_chain.invoke({"input": "Create a person named Alice, age 30"})
print(result.name)  # typed attribute access
```

### Branching Chains with `RunnableBranch`

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

# Route to different prompts based on input
branch = RunnableBranch(
    (lambda x: "technical" in x["topic"], technical_chain),
    (lambda x: "creative" in x["topic"], creative_chain),
    default_chain  # fallback
)
```

### Parallel Execution with `RunnableParallel`

```python
from langchain_core.runnables import RunnableParallel

# Run multiple chains at the same time
parallel = RunnableParallel(
    summary=summarize_chain,
    keywords=keywords_chain,
    sentiment=sentiment_chain,
)

result = parallel.invoke({"text": "Your document here..."})
print(result["summary"])    # all three run simultaneously
print(result["keywords"])
print(result["sentiment"])
```

---

## 12. Memory & Conversation History

LangChain provides several ways to persist conversation history across turns.

### In-Memory History (simple)

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Store history per session
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = prompt | llm | StrOutputParser()

# Wrap chain with history management
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# Each call with the same session_id automatically gets conversation history
response1 = chain_with_history.invoke(
    {"input": "My name is Alice"},
    config={"configurable": {"session_id": "user-123"}}
)

response2 = chain_with_history.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": "user-123"}}
    # LLM correctly answers "Alice" because history is injected
)
```

### Trimming Long Histories

```python
from langchain_core.messages import trim_messages

# Keep only the last N tokens to stay within context window
trimmed = trim_messages(
    messages,
    max_tokens=4000,
    strategy="last",           # keep the most recent messages
    token_counter=llm,         # use the LLM's tokenizer
    include_system=True,       # always keep the system message
)
```

---

## 13. Callbacks & Tracing

Callbacks let you observe and intercept every step of a chain or agent execution.

### Built-in Callback Handler

```python
from langchain_core.callbacks import StdOutCallbackHandler

# Prints every LLM call, tool use, and chain step to stdout
handler = StdOutCallbackHandler()

result = agent.invoke(
    {"messages": [HumanMessage(content="What is 5 * 6?")]},
    config={"callbacks": [handler]}
)
```

### Custom Callback Handler

```python
from langchain_core.callbacks import BaseCallbackHandler

class LoggingHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM called with: {prompts[0][:100]}...")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM responded: {response.generations[0][0].text[:100]}...")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool '{serialized['name']}' called with: {input_str}")

    def on_tool_end(self, output, **kwargs):
        print(f"Tool returned: {output}")

    def on_chain_error(self, error, **kwargs):
        print(f"Error: {error}")

# Attach to any chain or agent
result = chain.invoke(
    {"input": "Hello"},
    config={"callbacks": [LoggingHandler()]}
)
```

### LangSmith Tracing

LangSmith provides full trace visualization in the cloud:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# Now all chain/agent invocations are automatically traced
# View traces at: https://smith.langchain.com
```

---

## 14. How Everything Connects to LangGraph

Think of everything above as the **engine parts** — and LangGraph as the **chassis** that organizes them into a production vehicle.

| LangChain concept | Role in LangGraph |
|---|---|
| `init_chat_model()` | Creates the LLM node inside a graph |
| Message types (`HumanMessage`, `AIMessage`, etc.) | These **are** LangGraph's state — `MessagesState` is a typed dict holding `messages: list` |
| `@tool` / `StructuredTool` | Become **tool nodes** in the graph (`ToolNode`) |
| `create_react_agent()` | Is literally a LangGraph graph under the hood — it builds a `StateGraph` with an `agent` node and a `tools` node connected by a conditional edge |
| Manual agent loop | LangGraph replaces this with an explicit **graph** — each "loop iteration" is a node → edge → node traversal |
| ReAct pattern | LangGraph makes it explicit: the `should_continue` conditional edge checks `AIMessage.tool_calls` and routes to either the tool node or `END` |
| `.with_structured_output()` | Used inside LangGraph nodes to enforce typed outputs at each step |
| `RunnableWithMessageHistory` | Replaced by LangGraph's `checkpointer` for production persistence |
| Callbacks | LangGraph adds step-level streaming and human-in-the-loop interrupts |

### Minimal LangGraph Agent Example

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# 1. Define the agent node (calls the LLM)
def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# 2. Build the graph
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

# 3. Add edges
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    tools_condition,   # routes to "tools" if tool_calls exist, else END
)
graph.add_edge("tools", "agent")  # after tools, go back to agent

app = graph.compile()

# 4. Run
result = app.invoke({"messages": [HumanMessage(content="What is 99 * 42?")]})
```

The key shift: in plain LangChain, the loop is **implicit** (hidden inside `agent.invoke()`). In LangGraph, the loop is **explicit** — you draw it as a graph with nodes, edges, and conditional routing. This means you can:
- Add **human-in-the-loop** approval at any edge
- Persist state between runs (memory)
- Run multiple agents in parallel
- Branch into subgraphs
- Inspect and replay any step

---

## Quick Reference

### Import Cheatsheet

```python
# LLM initialization
from langchain.chat_models import init_chat_model

# Messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

# Tools
from langchain_core.tools import tool, StructuredTool, BaseTool

# Prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Output parsers
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

# Runnables (LCEL)
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda

# History
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Agents (LangGraph)
from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END
```

### Decision Guide — Which Tool to Use?

| Situation | Use |
|---|---|
| Simple function the LLM can call | `@tool` decorator |
| Function with rich parameter descriptions | `StructuredTool.from_function()` |
| Tool with custom state or async support | `BaseTool` subclass |
| Quick single-turn LLM call | `llm.invoke(messages)` |
| Multi-turn chat with history | `RunnableWithMessageHistory` |
| Typed/structured LLM output | `.with_structured_output(schema)` |
| Simple agent with tools | `create_react_agent()` |
| Production agent with persistence | LangGraph `StateGraph` |
| Compose prompt → LLM → parser | LCEL `|` pipe operator |

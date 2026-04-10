# 🚀 LangChain Core + Agentic Concepts (README)

This document explains the **core building blocks of LangChain Agents** using the modern ecosystem, focusing only on:

- Messages
- Chat Model Initialization
- Prompting
- Structured Output
- Tools & Tool Binding
- Tool Execution Flow
- Runnables (LCEL)
- Chains
- Agents (`create_agent`, `create_react_agent`)
- Observability (`@traceable`)

---

# 🧠 1. Messages (Foundation of Chat Systems)

LangChain uses structured message objects instead of plain text.

## 🔹 Types of Messages

### 1. `SystemMessage`

- Defines behavior of the assistant
- Acts like instructions

```python
SystemMessage(content="You are a helpful assistant")
```

---

### 2. `HumanMessage`

- Represents user input

```python
HumanMessage(content="What is AI?")
```

---

### 3. `AIMessage`

- Represents model response
- Contains:
  - `.content`
  - `.tool_calls` (important for agents)

---

### 4. `ToolMessage`

- Sent after a tool is executed
- Passed back to the LLM

```python
ToolMessage(content="42", tool_call_id="call_1")
```

---

# ⚙️ 2. Chat Model Initialization

## 🔹 `init_chat_model()`

Standard way to initialize models (provider-agnostic)

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq"
)
```

### ✅ Key Parameters

- `model` → model name
- `model_provider` → provider (groq, openai, etc.)

---

## 🔹 `.invoke()`

Used to call the model

```python
response = llm.invoke("Hello")
```

### Accepts:

- string input
- list of messages

### Returns:

- `AIMessage`

---

# 🧾 3. Prompting

## 🔹 Message-Based Prompting

```python
messages = [
    SystemMessage(content="Be concise"),
    HumanMessage(content="Explain Python")
]

llm.invoke(messages)
```

---

## 🔹 Why Important?

- Controls tone
- Controls format
- Guides tool usage

---

# 📦 4. Structured Output

## 🔹 `with_structured_output()`

Forces LLM to return structured data

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

structured_llm = llm.with_structured_output(User)
```

---

## 🔹 How it Works

- Converts prompt → JSON schema
- LLM must follow schema
- Output is validated automatically

---

# 🛠️ 5. Tools

## 🔹 `@tool` Decorator

Converts a Python function into a tool

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b
```

---

## 🔹 Important Rules

- Function name = tool name
- Docstring = tool description
- Type hints = input schema

---

# 🔗 6. Tool Binding

## 🔹 `bind_tools()`

Attaches tools to the model

```python
llm_with_tools = llm.bind_tools([add])
```

---

## 🔹 What Happens Internally?

- Model gets tool schema
- Model decides:
  - call tool OR
  - respond normally

---

## 🔹 `AIMessage.tool_calls`

After invoking:

```python
response = llm_with_tools.invoke("What is 2 + 3?")
```

You can inspect:

```python
response.tool_calls
```

---

# 🔁 7. Tool Execution Flow (Very Important)

## 🔹 Manual Loop

```python
ai_msg = llm_with_tools.invoke(messages)

if ai_msg.tool_calls:
    tool_call = ai_msg.tool_calls[0]

    result = add.invoke(tool_call["args"])

    tool_msg = ToolMessage(
        content=str(result),
        tool_call_id=tool_call["id"]
    )

    messages.append(ai_msg)
    messages.append(tool_msg)

    final = llm.invoke(messages)
```

---

## 🔹 Key Components

| Component       | Purpose           |
| --------------- | ----------------- |
| `tool_calls`    | LLM decision      |
| `tool.invoke()` | Executes tool     |
| `ToolMessage`   | Sends result back |
| Re-invoke LLM   | Final answer      |

---

# 🔄 8. Runnables (LCEL)

LangChain uses **Runnable interface** for pipelines.

---

## 🔹 `RunnablePassthrough`

Pass input unchanged

```python
from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough()
chain.invoke({"input": "hello"})
```

---

## 🔹 `RunnableSequence`

Chain steps sequentially

---

## 🔹 `RunnableParallel`

Run multiple operations at once

---

# 🔗 9. Chains

Chains = composition of runnables

## Example

```python
chain = prompt | llm
```

---

## 🔹 Benefits

- Modular design
- Reusable components
- Clean pipelines

---

# 🤖 10. Agents

Agents = LLM + Tools + Decision Loop

---

## 🔹 `create_agent()`

High-level API

```python
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[add],
    system_prompt="You are a math assistant"
)
```

---

### ✅ Features

- Automatic tool handling
- Minimal setup
- Best for quick apps

---

## 🔹 `create_react_agent()`

Mid-level control

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=[add]
)
```

---

### ✅ Features

- More control than `create_agent`
- Uses reasoning loop (ReAct)

---

## 🔹 Agent Invocation

```python
agent.invoke({
    "messages": [("user", "What is 5 + 5?")]
})
```

---

# 🔍 11. Observability

## 🔹 `@traceable`

Tracks execution in LangSmith

```python
from langsmith import traceable

@traceable
def run():
    return llm.invoke("Hello")
```

---

## 🔹 Why Use It?

- Debug prompts
- Track tool usage
- Monitor performance

---

# 🧠 Final Flow (Core Agent System)

```
Messages → Model → Tools → Tool Calls → Execution → Final Response
```

---

# ✅ Summary

You now understand:

- Message system
- Model initialization
- Prompting techniques
- Structured outputs
- Tool creation and binding
- Tool execution loop
- Runnable pipelines
- Chains
- Agent APIs
- Observability

---

# 🚀 Next Steps

- Build a chatbot using tools
- Integrate APIs as tools
- Add streaming
- Move to LangGraph (advanced)

---

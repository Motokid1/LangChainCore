# LangChain Core Concepts

## 1️⃣ ChatPromptTemplate vs PromptTemplate

- **PromptTemplate**: simple text-based prompts with single input.
- **ChatPromptTemplate**: structured for chat models; supports multiple messages (system, human, AI).

## 2️⃣ Variables in Templates (`{}`)

- `{variable}` inside a template must be provided at invoke time.
- Missing variables cause `KeyError`.
- Raw JSON inside templates with `{}` is interpreted as variables unless handled properly.
- LangChain always expects inputs as a dictionary, even if there are no variables
- LangChain ALWAYS passes data as a dictionary between steps

## 3️⃣ get_format_instructions()

- Returns instructions for structured output from an OutputParser (e.g., Pydantic).
- Ensures model output is valid JSON or structured data.
- Automatically enforces schema for safe parsing.

## 4️⃣ partial()

- Pre-fills template variables before `invoke()`.
- Safely inserts large JSON instructions without KeyErrors.
- Makes templates reusable and maintainable.

## 5️⃣ PydanticOutputParser

- Converts model output JSON → Python object using Pydantic.
- Easy access to fields: `result.name`, `result.age`.
- Fails if model output is invalid JSON.

## 6️⃣ Direct JSON in Template vs Variable + partial()

- Direct JSON often causes KeyError due to `{}`.
- Using `{format_instructions}` + `partial()` is safe and clean.

## 7️⃣ Chains (`|` operator)

- Chains connect prompt → LLM → parser → memory (optional).
- Simplifies data flow between components.
- Useful for pipelines like: prompt → model → parse → store.

## 8️⃣ Memory

- Stores previous conversation/messages for context-aware responses.
- Types: `ConversationBufferMemory` (raw messages), `ConversationSummaryMemory` (summarized context).
- Makes chatbots stateful.

## 9️⃣ Passing Variables in invoke() Instead of partial()

- Works if all variables are passed manually.
- Extremely error-prone and hard to maintain.
- Only recommended for very small/simple prompts.

## 10️⃣ Best Practices

- Always use `partial()` for JSON or large instruction blocks.
- Use `get_format_instructions()` for structured output.
- Use `PydanticOutputParser` or other parsers for typed results.
- Keep chains modular: prompt → LLM → parser → memory.
- Avoid manually passing all variables in complex templates.

---

# LangChain Core - Runnables Explained

## 1️⃣ What is a Runnable?

- Any object that can be invoked with input and produces output.
- Examples: PromptTemplate, LLM, OutputParser, Python function.
- All runnables implement `.invoke(input)` or `.batch(inputs)`.
- Think of it as a **step in your pipeline**.

## 2️⃣ Runnable

- Base class for all runnables.
- Wraps any callable.

```python
from langchain_core.runnables import Runnable

def double_number(x):
    return x * 2

r = Runnable(double_number)
print(r.invoke(5))  # Output: 10
```

## 3️⃣ LambdaRunnable

- Shortcut to wrap a simple function or lambda.

```python
from langchain_core.runnables import LambdaRunnable

r = LambdaRunnable(lambda x: x + 10)
print(r.invoke(5))  # Output: 15
```

- Useful for preprocessing or postprocessing in a chain.

## 4️⃣ RunnableSequence

- Runs a **list of Runnables sequentially**.
- Output of one is input to the next.

```python
from langchain_core.runnables import RunnableSequence, LambdaRunnable

r1 = LambdaRunnable(lambda x: x * 2)
r2 = LambdaRunnable(lambda x: x + 5)

seq = RunnableSequence([r1, r2])
print(seq.invoke(3))  # Output: 11
```

- Handy for pipelines like: Prompt → LLM → Parser.

## 5️⃣ RunnableMap

- Runs **multiple Runnables in parallel**, same input.
- Returns a dictionary of results.

```python
from langchain_core.runnables import RunnableMap, LambdaRunnable

map_r = RunnableMap({
    "double": LambdaRunnable(lambda x: x*2),
    "triple": LambdaRunnable(lambda x: x*3)
})

print(map_r.invoke(5))  # Output: {'double': 10, 'triple': 15}
```

- Useful for generating multiple outputs from same input.

## 6️⃣ RunnableParallel

- Like RunnableMap but **concurrent execution**.
- Good for slow tasks (like multiple LLM calls).

```python
from langchain_core.runnables import RunnableParallel, LambdaRunnable

parallel = RunnableParallel([
    LambdaRunnable(lambda x: x*2),
    LambdaRunnable(lambda x: x+10)
])

print(parallel.invoke(5))  # Output: [10, 15]
```

## 7️⃣ Summary Table

| Runnable Type    | Purpose                             | Input → Output                 | When to Use                       |
| ---------------- | ----------------------------------- | ------------------------------ | --------------------------------- |
| Runnable         | Wrap any callable                   | 1:1                            | Simple reusable steps             |
| LambdaRunnable   | Wrap a lambda or small function     | 1:1                            | Small transformations             |
| RunnableSequence | Chain multiple runnables            | Output of prev → Input of next | Pipelines (Prompt → LLM → Parser) |
| RunnableMap      | Run multiple runnables in parallel  | Same input → dict of outputs   | Multi-output pipelines            |
| RunnableParallel | Run multiple runnables concurrently | Same input → list of outputs   | Speed up slow tasks               |

## Key Takeaways

- Everything in LangChain pipeline is a Runnable.
- `|` operator = shortcut for RunnableSequence.
- RunnableMap & RunnableParallel = branch or parallelize pipelines.
- LambdaRunnable = simple way to inject Python logic anywhere.

# LangChain Core - Memory Explained

## 1️⃣ What is Memory in LangChain?

- Memory is how LangChain **remembers information across steps or conversations**.
- Think of it as **state** in your pipeline or chatbot.
- Without memory: every prompt is **stateless** → model forgets previous messages.
- With memory: the model can **use context** to make informed responses.

## 2️⃣ Why Memory is Useful

- Chatbots can **maintain conversation history**.
- Pipelines can **reuse intermediate results**.
- Useful for **personalization** (user name, preferences).
- Enables **dynamic chains** that adapt based on prior inputs.

## 3️⃣ Types of Memory

### a) ConversationBufferMemory

- Stores all messages **as-is**.
- Useful for **simple chatbots**.

```python
from langchain_core.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.add_user_message("Hi, my name is Alex")
memory.add_ai_message("Hello Alex! How can I help you today?")

print(memory.load_memory_variables({}))
# Output: {'history': 'Human: Hi, my name is Alex\nAI: Hello Alex! How can I help you today?'}
```

- **Pros**: Simple, preserves raw messages.
- **Cons**: History can grow indefinitely → large context → costly in LLM calls.

### b) ConversationSummaryMemory

- Stores a **summary of conversation** instead of full messages.
- Keeps memory **compact**, suitable for long chats.

```python
from langchain_core.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm, max_token_limit=50)
memory.add_user_message("I love Python and AI")
memory.add_ai_message("Great! What project are you working on?")

print(memory.load_memory_variables({}))
# Output: {'history': 'User likes Python and AI. They are discussing projects.'}
```

- **Pros**: Reduces context size → cheaper and faster LLM calls.
- **Cons**: Summaries may lose fine-grained info.

### c) Custom Memory

- Implement your own memory logic by **subclassing `BaseMemory`**.
- Example: store **user preferences** or **form data**.

## 4️⃣ How Memory Fits in a Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

# Setup memory
memory = ConversationBufferMemory()

# Example chain
chain = RunnableSequence([
    prompt,   # prompt template
    llm,      # language model
    parser    # output parser
])

# Add previous conversation
memory.add_user_message("Hello!")
memory.add_ai_message("Hi there!")

# Load memory into prompt variables
input_vars = {"history": memory.load_memory_variables({})['history']}
output = chain.invoke(input_vars)

print(output)
```

- Memory variables (like `{history}`) are injected into prompts dynamically.
- Keeps context across multiple `invoke()` calls.

## 5️⃣ Key Concepts

1. **Memory Variables** → placeholders in prompts that store history or summary.
2. **Stateful vs Stateless** → with memory, chains become **stateful**.
3. **History Growth** → manage size via summary or token limit.
4. **Integration** → memory is optional, can be added to any chain.

## 6️⃣ Best Practices

- Use `ConversationBufferMemory` for **short chats** or testing.
- Use `ConversationSummaryMemory` for **long-term memory**.
- Always **inject memory variables into prompt** using `{history}`.
- Be careful with **memory size** → long conversations → higher LLM cost.

## 💡 Analogy for Beginners

- Memory is like a **sticky note** your chatbot keeps:
  - **Buffer memory** → writes every word you say.
  - **Summary memory** → writes a short note summarizing what you said.

# LangChain Core Deep Explanation

## 🔁 Why RunnableSequence Expects Runnables

### ✅ Core Idea

`RunnableSequence` is designed to execute steps **sequentially**, where:

- Each step takes the output of the previous step
- The flow must be **linear and unambiguous**

### 🔄 Execution Flow

Input → Step1 → Step2 → Step3 → Output

### 🧠 Why Only Runnables?

Each step must be:

- Executable
- Accept input
- Return output

So LangChain enforces:

- Runnable
- Callable (auto-converted to Runnable)

### ❌ Why NOT list?

```python
RunnableSequence([step1, step2])
```

Problems:

- Internally may treat it as a single object
- Causes ambiguity
- Version-dependent behavior

### ❌ Why NOT dict?

```python
RunnableSequence({
  "a": step1,
  "b": step2
})
```

Problems:

- No execution order
- Output mapping unclear
- Breaks linear pipeline concept

### ✅ Correct Usage

```python
RunnableSequence(step1, step2, step3)
```

---

## ⚡ Why RunnableParallel Uses Dictionary

### ✅ Core Idea

`RunnableParallel` runs multiple steps **simultaneously**

### 🔀 Execution Flow

Input →
→ Step A
→ Step B
→ Step C

### 🧠 Why Dictionary?

Because:

- Each output must have a **name**
- Results must be structured

Example output:

```python
{
  "definition": result1,
  "example": result2
}
```

### ❌ Why NOT list?

```python
RunnableParallel([step1, step2])
```

Problems:

- No labels
- Output becomes:

```python
[result1, result2]
```

- Hard to identify which is which

---

## 📤 Output Behavior Difference

### RunnableSequence Output

Returns:

```python
AIMessage
```

So:

```python
response.content
```

✅ Works

---

### RunnableParallel Output

Returns:

```python
dict
```

Example:

```python
{
  "definition": AIMessage,
  "example": AIMessage
}
```

So:

```python
result.content
```

❌ Fails

---

## ❌ Why `.content` Fails in Parallel

Because:

- `result` is a dictionary
- Dictionary does NOT have `.content`

---

## ✅ Correct Way to Access

```python
result["definition"].content
result["example"].content
```

---

## ⚠️ When Parser is Used

If you use:

```python
RunnableLambda(...) | llm | parser
```

Then output becomes:

- dict
- or Pydantic object
- or structured data

### Example:

```python
result["definition"]
```

Now:

- `.content` may NOT exist

---

## 🧠 Why Parser Helps

Parser converts raw LLM output into:

- structured format
- predictable schema

So instead of:

```python
AIMessage.content
```

You get:

```python
{
  "name": "Alex",
  "age": 25
}
```

---

## 🔥 Key Differences

| Feature     | RunnableSequence | RunnableParallel |
| ----------- | ---------------- | ---------------- |
| Execution   | Sequential       | Parallel         |
| Input       | Arguments        | Dictionary       |
| Output      | Single value     | Dict             |
| `.content`  | Works            | Not directly     |
| Parser need | Optional         | Recommended      |

---

## 🧠 Easy Memory Trick

Sequence → Pipeline (➡️➡️➡️)  
Parallel → Branches (🔀)

---

## 🚀 Final Takeaways

- Sequence needs **ordered steps → Runnable only**
- Parallel needs **named outputs → Dictionary**
- `.content` works only when output is `AIMessage`
- Use parser for clean structured outputs in parallel

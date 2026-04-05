# LangChain RAG System — Complete Theory & Reference Guide

> A structured deep-dive into building a Retrieval-Augmented Generation (RAG) system using LangChain Core, covering every component from document loading to memory-aware conversations.

---

## Table of Contents

1. [What is RAG and Why Use It?](#1-what-is-rag-and-why-use-it)
2. [Environment & LLM Setup](#2-environment--llm-setup)
3. [Document Loading](#3-document-loading)
4. [Text Splitting](#4-text-splitting)
5. [Embeddings](#5-embeddings)
6. [Vector Store & Persistent Storage](#6-vector-store--persistent-storage)
7. [Prompt Templates](#7-prompt-templates)
8. [Formatting Retrieved Documents](#8-formatting-retrieved-documents)
9. [LCEL RAG Chain](#9-lcel-rag-chain)
10. [Memory & Multi-Turn Conversation](#10-memory--multi-turn-conversation)
11. [Running the Chat Loop](#11-running-the-chat-loop)
12. [Best Practices & Key Takeaways](#12-best-practices--key-takeaways)
13. [Full Pipeline at a Glance](#13-full-pipeline-at-a-glance)

---

## 1. What is RAG and Why Use It?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances an LLM's responses by injecting relevant, retrieved context from your own documents into each prompt — before the model generates a reply.

### The Problem RAG Solves

LLMs are trained on a fixed dataset with a knowledge cutoff. They:

- Don't know your private documents
- Can hallucinate when asked about specific, niche, or recent information
- Have a limited context window that can't hold thousands of pages

### How RAG Works

```
User Question
     │
     ▼
[Retriever] ──── searches vector store ────► [Top-K Relevant Chunks]
     │                                                  │
     └───────────────────────────────────────────────► │
                                                        ▼
                                              [Prompt with Context]
                                                        │
                                                        ▼
                                                [LLM generates answer]
```

RAG gives the LLM the **right information at the right time**, grounding its answers in your data.

---

## 2. Environment & LLM Setup

### Theory

Before doing anything, we need:

1. A secure way to manage API keys (never hardcode them)
2. An initialized LLM to use throughout the pipeline

### `dotenv` — Loading API Keys Securely

Environment variables are stored in a `.env` file and loaded at runtime. This keeps secrets out of your source code and version control.

```python
from dotenv import load_dotenv
load_dotenv()  # Reads .env file and populates os.environ
```

**`.env` file example:**

```
HF_TOKEN=hf_xxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

**Why:** Never hardcode API keys. If committed to git, they can be compromised permanently.

### `ChatGroq` — Initializing the LLM

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-120b")
```

`ChatGroq` wraps the Groq inference API, which gives fast access to large open-source models. It follows LangChain's standard `BaseChatModel` interface, so it integrates cleanly with chains, prompts, and memory.

**Why Groq?** Extremely low-latency inference — ideal for interactive chat applications.

---

## 3. Document Loading

### Theory

Before we can retrieve anything, we must load our source documents into memory as LangChain `Document` objects — each containing `page_content` (text) and `metadata` (source path, page number, etc.).

### `DirectoryLoader` — Load Files From a Folder

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

txt_loader = DirectoryLoader(
    "data/",
    glob="**/*.txt",        # Matches all .txt files recursively
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = txt_loader.load()
```

| Parameter       | Purpose                              |
| --------------- | ------------------------------------ |
| `path`          | Root directory to scan               |
| `glob`          | File pattern (e.g., `**/*.pdf`)      |
| `loader_cls`    | Which loader class to use per file   |
| `loader_kwargs` | Extra arguments passed to the loader |

### `PyPDFLoader` — Load PDFs

```python
pdf_loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs.extend(pdf_loader.load())
```

PDF loading can be tricky due to encoding differences, scanned pages, or multi-column layouts. `PyPDFLoader` handles most standard PDFs.

**Best Practice:** Always guard against empty document sets:

```python
if not docs:
    raise ValueError("No documents found in data/ folder!")
```

---

## 4. Text Splitting

### Theory

LLMs have a **context window limit** (e.g., 8K or 32K tokens). A single document may be thousands of words long — it can't fit into a prompt directly. We split documents into smaller **chunks** that can be embedded individually and retrieved selectively.

Splitting strategy matters:

- **Too large:** Chunks may exceed context limits or carry irrelevant noise
- **Too small:** You lose the surrounding context needed to understand a fact
- **Overlap:** A small overlap between chunks ensures sentences at boundaries aren't cut off mid-thought

### `RecursiveCharacterTextSplitter`

This is the recommended general-purpose splitter. It tries to split on natural boundaries in order: `\n\n` (paragraphs) → `\n` (lines) → `.` (sentences) → ` ` (words) — falling back to harder splits only when needed.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # Max characters per chunk
    chunk_overlap=50    # Characters shared between adjacent chunks
)
split_docs = splitter.split_documents(docs)

# Remove empty chunks
split_docs = [doc for doc in split_docs if doc.page_content.strip()]
```

**Why `strip()` filter?** Loaders sometimes produce chunks with only whitespace. These create empty, useless embeddings that pollute the vector store.

**Best Practice:**

```python
if not split_docs:
    raise ValueError("No valid chunks after splitting!")
```

---

## 5. Embeddings

### Theory

An **embedding** is a dense numerical vector (e.g., 384 numbers) that represents the semantic meaning of a piece of text. Texts with similar meanings have embeddings that are geometrically close in vector space.

This allows us to search for semantically relevant chunks — not just keyword matches.

```
"What is photosynthesis?"  →  [0.12, -0.34, 0.87, ...]  (384-dim vector)
"Plants convert light to energy"  →  [0.11, -0.31, 0.85, ...]  (nearby!)
"Recipe for chocolate cake"  →  [-0.72, 0.44, -0.22, ...]  (far away)
```

### `HuggingFaceEmbeddings`

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

**Why `all-MiniLM-L6-v2`?**

- Free, no API key needed
- Runs locally (no data leaves your machine)
- Compact (22M parameters) yet highly effective for semantic search
- Produces 384-dimensional vectors — fast to compute and store

**Why not use the LLM itself for embeddings?** Embedding models are specialized for producing comparable vectors. LLMs are optimized for generation, not similarity comparison.

---

## 6. Vector Store & Persistent Storage

### Theory

A **vector store** is a database optimized for storing embeddings and performing fast **similarity search** — given a query vector, find the N most similar stored vectors and return their associated documents.

### `Chroma` — Local Persistent Vector Store

```python
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(
    split_docs,             # Documents to embed and store
    embeddings,             # Embedding model to use
    persist_directory="./chroma_db"  # Save to disk
)
```

`Chroma.from_documents` does three things:

1. Runs each chunk through the embedding model
2. Stores the vectors in Chroma's internal database
3. Persists everything to `./chroma_db` so you don't re-embed on next run

### Creating a Retriever

```python
retriever = db.as_retriever(search_kwargs={"k": 3})
```

`as_retriever()` wraps the vector store in LangChain's standard `BaseRetriever` interface. The `k=3` setting means: for each query, return the 3 most semantically similar chunks.

**Why persist?** Embedding thousands of documents is slow and costs compute. Persistence means you embed once and retrieve forever.

**Best Practice:** Load from existing store when available:

```python
# Re-use existing store instead of re-embedding
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
```

---

## 7. Prompt Templates

### Theory

A **prompt template** is a reusable, parameterized structure for building LLM inputs. Instead of writing the same prompt logic repeatedly, you define placeholders like `{context}`, `{input}`, and `{chat_history}` that get filled at runtime.

### `ChatPromptTemplate`

Chat models expect a sequence of messages (system, human, assistant), not a single string. `ChatPromptTemplate.from_messages()` builds this structure.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use the retrieved context to answer the question.
If the answer is not found, say "I don't know."
Include source references at the end.

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),  # Injects full conversation history
    ("human", "{input}")                  # Current user question
])
```

| Component                             | Purpose                                                     |
| ------------------------------------- | ----------------------------------------------------------- |
| `("system", ...)`                     | Sets LLM behavior and injects retrieved context             |
| `MessagesPlaceholder("chat_history")` | Dynamically inserts previous messages for multi-turn memory |
| `("human", "{input}")`                | The user's current question                                 |

**Why `MessagesPlaceholder`?** Unlike a simple `{chat_history}` string, this properly handles the list of `HumanMessage`/`AIMessage` objects that LangChain memory produces.

---

## 8. Formatting Retrieved Documents

### Theory

The retriever returns a list of `Document` objects. The LLM expects a single string as context. We need a function to bridge this gap — and ideally include **source attribution** so the LLM can cite where it found the answer.

```python
def format_docs(docs):
    return "\n\n".join(
        f"{doc.page_content}\nSOURCE: {doc.metadata.get('source', 'unknown')}"
        for doc in docs
    )
```

**What it does:**

- Joins all retrieved chunks with blank lines between them
- Appends each chunk's file path from `doc.metadata['source']`
- Falls back to `"unknown"` if metadata is missing

**Why include source?** When the prompt contains `SOURCE: data/report.pdf`, the LLM can reference it in its answer. This makes the system **auditable** — users can verify where information came from.

---

## 9. LCEL RAG Chain

### Theory

**LCEL (LangChain Expression Language)** is a declarative, functional way to compose chains using the pipe operator (`|`). Each component receives input, transforms it, and passes it to the next.

The RAG chain must:

1. Extract the query string from the input dict
2. Pass it to the retriever
3. Format the retrieved docs
4. Build a prompt with context + chat history + query
5. Send to the LLM
6. Parse the output as a string

### The Chain

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def get_input_str(x):
    """Extract the query string from dict input."""
    return x["input"] if isinstance(x, dict) else x

rag_chain = (
    {
        # Parallel: retrieves context while passing other values through
        "context": RunnableLambda(get_input_str) | retriever | format_docs,
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", [])
    }
    | qa_prompt          # Fills in the template
    | llm                # Generates a response
    | StrOutputParser()  # Extracts the text string from the AIMessage
)
```

### Component Breakdown

| Component                       | Type                | Role                                      |
| ------------------------------- | ------------------- | ----------------------------------------- |
| `RunnableLambda(get_input_str)` | Runnable            | Extracts `"input"` key from dict          |
| `retriever`                     | BaseRetriever       | Returns top-k matching `Document` objects |
| `format_docs`                   | Function → Runnable | Converts docs to context string           |
| `qa_prompt`                     | ChatPromptTemplate  | Builds the full message list              |
| `llm`                           | BaseChatModel       | Generates the response                    |
| `StrOutputParser()`             | OutputParser        | Extracts `.content` from `AIMessage`      |

**Why the `get_input_str` wrapper?** When `RunnableWithMessageHistory` is used (next section), the chain receives a dict like `{"input": "...", "chat_history": [...]}`. The retriever expects a plain string — this function handles the conversion.

**Why `StrOutputParser`?** The LLM returns an `AIMessage` object. `StrOutputParser` extracts its `.content` string, making the chain's output directly usable.

---

## 10. Memory & Multi-Turn Conversation

### Theory

By default, each chain invocation is stateless — the LLM has no memory of previous turns. For a chat application, we need to maintain **conversation history** per session and inject it into each prompt.

LangChain solves this with two components:

- **`ChatMessageHistory`** — Stores the list of messages for a session
- **`RunnableWithMessageHistory`** — Wraps any chain, automatically loading/saving history

### Store and Session Management

```python
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}  # In-memory store: { session_id: ChatMessageHistory }

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

This function is a **factory** — it creates a new history for new sessions and returns the existing one for returning sessions.

**Why a dict keyed by session_id?** This allows multiple users (or conversations) to each have their own independent history, running concurrently.

### Wrapping the Chain with Memory

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

rag_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",       # Key for current user message
    history_messages_key="chat_history"  # Key for injected history
)
```

Under the hood, `RunnableWithMessageHistory`:

1. Calls `get_session_history(session_id)` to load history
2. Injects it as `chat_history` into the chain input
3. Appends the new human message and AI response to history after each turn

**Why not manually manage history?** This approach is automatic, thread-safe per session, and cleanly separates state management from chain logic.

---

## 11. Running the Chat Loop

### Theory

The interactive loop ties everything together. It reads user input, invokes the memory-aware chain with a session ID, and prints the response.

```python
print("\n✅ RAG System Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    response = rag_with_memory.invoke(
        {"input": query},
        config={"configurable": {"session_id": "user1"}}
    )

    print("\nAI:", response, "\n")
```

### Key Parameters

| Parameter                                          | Purpose                                   |
| -------------------------------------------------- | ----------------------------------------- |
| `{"input": query}`                                 | The current user message                  |
| `config={"configurable": {"session_id": "user1"}}` | Identifies which session's history to use |

**Why `session_id`?** It's the key that `get_session_history` uses to look up or create a `ChatMessageHistory`. Change the value to support multiple simultaneous users.

---

## 12. Best Practices & Key Takeaways

### Security

- ✅ Always use `.env` + `load_dotenv()` for API keys
- ✅ Add `.env` to `.gitignore` — never commit secrets

### Document Loading

- ✅ Handle encoding explicitly: `loader_kwargs={"encoding": "utf-8"}`
- ✅ Use `try/except` around PDF loading (corrupted files are common)
- ✅ Validate that documents were actually loaded before proceeding

### Chunking

- ✅ `chunk_size=500` with `chunk_overlap=50` is a solid default; tune per domain
- ✅ Always filter empty chunks with `.strip()`
- ✅ Larger overlap = better context continuity, but higher storage cost

### Embeddings & Vector Store

- ✅ Use `all-MiniLM-L6-v2` for free, local, high-quality embeddings
- ✅ Persist your Chroma DB — never re-embed on every run
- ✅ `k=3` is a good starting point; increase if answers are incomplete

### Prompting

- ✅ Always include a clear system instruction (behavior + fallback)
- ✅ Use `MessagesPlaceholder` for chat history — not raw string formatting
- ✅ Include source attribution in context to enable citation in answers

### Chain Design

- ✅ Use `RunnableLambda` to adapt between dict and string interfaces
- ✅ Use `StrOutputParser` to get clean string output, not `AIMessage` objects
- ✅ Keep the chain modular — each step should do one thing

### Memory

- ✅ Always use `session_id` even in single-user apps — makes scaling easy later
- ✅ For production, replace `InMemoryChatMessageHistory` with a persistent backend (Redis, PostgreSQL)
- ✅ Consider trimming history after N turns to prevent context overflow

---

## 13. Full Pipeline at a Glance

```
[.env file]
    │  load_dotenv()
    ▼
[ChatGroq LLM]

[data/ folder]
    │  DirectoryLoader + TextLoader / PyPDFLoader
    ▼
[Raw Documents]  →  list of Document(page_content, metadata)
    │  RecursiveCharacterTextSplitter(chunk_size=500, overlap=50)
    ▼
[Chunks]  →  smaller Document objects
    │  HuggingFaceEmbeddings("all-MiniLM-L6-v2")
    ▼
[Vectors]
    │  Chroma.from_documents(persist_directory="./chroma_db")
    ▼
[Vector Store]  ←──────────────────────────────────────┐
    │  as_retriever(k=3)                                │
    ▼                                                   │
[Retriever]                                             │
                                                        │
[User Query]                                            │
    │                                                   │
    ├── RunnableLambda ──► [Retriever] ──► format_docs ─┘
    │                                         │
    ├── chat_history ◄── RunnableWithMessageHistory
    │                                         │
    └──────────────────── qa_prompt ◄─────────┘
                              │
                             [LLM]
                              │
                        StrOutputParser
                              │
                           [Response]
                              │
                    Saved back to ChatMessageHistory
```

---

## Quick Reference: All Classes & Functions

| Module                                       | Class / Function                     | What It Does                       |
| -------------------------------------------- | ------------------------------------ | ---------------------------------- |
| `dotenv`                                     | `load_dotenv()`                      | Loads `.env` into `os.environ`     |
| `langchain_groq`                             | `ChatGroq(model=...)`                | Initializes the LLM                |
| `langchain_community.document_loaders`       | `DirectoryLoader`                    | Scans a folder and loads files     |
| `langchain_community.document_loaders`       | `TextLoader`                         | Loads a `.txt` file                |
| `langchain_community.document_loaders`       | `PyPDFLoader`                        | Loads a `.pdf` file                |
| `langchain_text_splitters`                   | `RecursiveCharacterTextSplitter`     | Splits documents into chunks       |
| `langchain_huggingface`                      | `HuggingFaceEmbeddings`              | Converts text to embedding vectors |
| `langchain_community.vectorstores`           | `Chroma.from_documents()`            | Embeds and stores docs in Chroma   |
| `langchain_community.vectorstores`           | `db.as_retriever()`                  | Creates a retriever from the store |
| `langchain_core.prompts`                     | `ChatPromptTemplate.from_messages()` | Builds a multi-message prompt      |
| `langchain_core.prompts`                     | `MessagesPlaceholder`                | Dynamic slot for message history   |
| `langchain_core.runnables`                   | `RunnableLambda`                     | Wraps any function as a Runnable   |
| `langchain_core.output_parsers`              | `StrOutputParser`                    | Extracts string from LLM output    |
| `langchain_community.chat_message_histories` | `ChatMessageHistory`                 | Stores conversation messages       |
| `langchain_core.runnables.history`           | `RunnableWithMessageHistory`         | Wraps chain with session memory    |

---

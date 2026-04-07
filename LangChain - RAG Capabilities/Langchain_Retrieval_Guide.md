# LangChain Retrieval — Complete Theory & Reference Guide

> A deep-dive into retrieval concepts in LangChain: `similarity_search`, `as_retriever`, `RetrievalQA`, `BaseRetriever`, and production-grade best practices.

---

## Table of Contents

1. [similarity_search](#1-similarity_search)
2. [as_retriever](#2-as_retriever)
3. [similarity_search vs as_retriever — Comparison](#3-similarity_search-vs-as_retriever--comparison)
4. [RetrievalQA](#4-retrievalqa)
5. [Embeddings Rate Limiting](#5-embeddings-rate-limiting)
6. [BaseRetriever — The Core Abstraction](#6-baseretriever--the-core-abstraction)
7. [Building a Custom Retriever](#7-building-a-custom-retriever)
8. [How Everything Connects](#8-how-everything-connects)
9. [Best Practices & Key Takeaways](#9-best-practices--key-takeaways)
10. [Complete RAG Mental Model](#10-complete-rag-mental-model)

---

## 1. `similarity_search`

### What It Is

`similarity_search` is a **direct method on the vector store object** itself. It lets you manually query the database and get back the most similar documents to your query — no pipeline, no abstraction, just raw retrieval.

### How It Works

When you call `similarity_search`, the vector store:

1. Embeds your query string into a vector
2. Computes cosine (or dot-product) similarity between the query vector and all stored vectors
3. Returns the top-`k` most similar documents

```python
docs = db.similarity_search("What is RAG?", k=3)

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
```

### Key Points

- Returns a plain Python `list` of `Document` objects
- You are responsible for everything after this — prompting, LLM calls, formatting
- Not a `Runnable` — cannot be plugged into a `|` chain directly
- Gives you **maximum visibility** into what's being retrieved

### When to Use It

- Debugging your vector store (are the right chunks being retrieved?)
- Exploring what's stored before building a full chain
- One-off retrieval without needing a pipeline
- Learning how semantic search works under the hood

---

## 2. `as_retriever`

### What It Is

`as_retriever()` is a method on any LangChain vector store that **wraps it in a `BaseRetriever`-compatible object** — making it a proper `Runnable` that plugs directly into LCEL chains.

### How It Works

```python
retriever = db.as_retriever(
    search_kwargs={"k": 3}   # Return top 3 matching chunks
)

# Use standalone
docs = retriever.invoke("What is RAG?")

# Or pipe directly into a chain
chain = retriever | format_docs | prompt | llm | StrOutputParser()
```

### Supported Options

```python
retriever = db.as_retriever(
    search_type="similarity",          # Default — cosine similarity
    # search_type="mmr",               # Max Marginal Relevance — reduces redundancy
    # search_type="similarity_score_threshold",  # Filter by minimum score
    search_kwargs={
        "k": 5,                        # Number of docs to return
        "filter": {"source": "report.pdf"},  # Metadata filtering
        "score_threshold": 0.7         # Used with similarity_score_threshold
    }
)
```

### Key Points

- Returns a `BaseRetriever` — a full `Runnable` object
- Supports `.invoke()`, `.batch()`, `.stream()`
- Works seamlessly with the `|` pipe operator in LCEL
- Abstracts away the embedding + similarity logic

### When to Use It

- Building any LCEL RAG pipeline (which is almost always)
- When you need metadata filtering or different search strategies
- Any production system where retrieval is part of a larger chain

---

## 3. `similarity_search` vs `as_retriever` — Comparison

| Feature             | `similarity_search`    | `as_retriever`                   |
| ------------------- | ---------------------- | -------------------------------- |
| Type                | Direct DB method       | Runnable abstraction             |
| Returns             | `list[Document]`       | `BaseRetriever` (Runnable)       |
| LCEL compatible     | ❌ No                  | ✅ Yes                           |
| Pipeable with `\|`  | ❌ No                  | ✅ Yes                           |
| Control level       | High — manual          | Medium — configured              |
| Search type options | Limited                | `similarity`, `mmr`, `threshold` |
| Metadata filtering  | Basic                  | Full support via `search_kwargs` |
| Best for            | Debugging, exploration | Production RAG pipelines         |

**One-line summary:**

- `similarity_search` = manual retrieval, great for inspecting results
- `as_retriever` = pipeline-ready retrieval, great for building chains

---

## 4. `RetrievalQA`

### What It Is

`RetrievalQA` is a **legacy prebuilt chain** in LangChain that bundles a retriever, a prompt, and an LLM together into a single object. It was the standard way to build RAG before LCEL was introduced.

### How It Works

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",      # "stuff" = put all docs into one prompt
    retriever=retriever
)

response = qa.invoke({"query": "What is RAG?"})
print(response["result"])
```

### Chain Types

| `chain_type`   | Behavior                                     | Use When                       |
| -------------- | -------------------------------------------- | ------------------------------ |
| `"stuff"`      | All docs stuffed into one prompt             | Few, short documents           |
| `"map_reduce"` | Each doc processed separately, then combined | Many or large documents        |
| `"refine"`     | Iteratively refines the answer doc by doc    | Need progressive improvement   |
| `"map_rerank"` | Scores each doc's answer, picks best         | Need highest-confidence answer |

### Limitations

- Not LCEL-native — doesn't use the `|` pipe operator
- Difficult to customize prompts
- Memory integration is clunky compared to `RunnableWithMessageHistory`
- Less transparent — harder to debug intermediate steps
- Considered **legacy** — Anthropic recommends LCEL for new projects

### When to Use It

- Quick prototypes where you just want something working in 3 lines
- Beginner experiments and learning
- Avoid in production — use a custom LCEL chain instead

---

## 5. Embeddings Rate Limiting

### The Problem

When using hosted embedding APIs or even local models under heavy load, you may encounter:

- HTTP 429 (rate limit) errors
- Timeout errors
- Connection resets

If your embedding call fails mid-way through processing hundreds of documents, you lose all progress.

### Solution: Retry Wrapper

Wrap your embeddings in a class that catches errors and retries with a delay:

```python
from langchain_huggingface import HuggingFaceEmbeddings
import time

class SafeEmbeddings:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.retry_seconds = 10

    def embed_documents(self, texts):
        """Called when embedding a list of chunks (e.g., during Chroma.from_documents)."""
        while True:
            try:
                return self.embeddings.embed_documents(texts)
            except Exception as e:
                print(f"Embedding error: {e}")
                print(f"Retrying in {self.retry_seconds}s...")
                time.sleep(self.retry_seconds)

    def embed_query(self, text):
        """Called when embedding a single query at retrieval time."""
        while True:
            try:
                return self.embeddings.embed_query(text)
            except Exception as e:
                print(f"Embedding error: {e}")
                print(f"Retrying in {self.retry_seconds}s...")
                time.sleep(self.retry_seconds)
```

### Using It

```python
embeddings = SafeEmbeddings()

# Drop-in replacement — Chroma calls embed_documents() and embed_query() internally
db = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
```

### Why Two Methods?

LangChain's vector stores call different methods depending on the context:

| Method                   | When called                      | Input       |
| ------------------------ | -------------------------------- | ----------- |
| `embed_documents(texts)` | During indexing (storing chunks) | `list[str]` |
| `embed_query(text)`      | During retrieval (searching)     | `str`       |

Both must be present for the wrapper to work as a drop-in replacement.

### Production Enhancement: Exponential Backoff

For more robust production code, increase the wait time after each failure:

```python
def embed_with_backoff(self, fn, *args):
    wait = 5
    for attempt in range(5):
        try:
            return fn(*args)
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
            wait *= 2  # Double the wait each time: 5, 10, 20, 40, 80
    raise RuntimeError("Embedding failed after 5 attempts")
```

---

## 6. `BaseRetriever` — The Core Abstraction

### What It Is

`BaseRetriever` is the **abstract base class** that all LangChain retrievers inherit from. It defines the standard interface that every retriever — whether backed by Chroma, FAISS, BM25, Wikipedia, or your own custom logic — must follow.

Think of it as a contract:

> _"Whatever retrieval strategy you use, you must be callable with `.invoke(query)` and return `List[Document]`."_

### Why It Exists

Without `BaseRetriever`, every vector store would have a different method name and return type. You'd have to write different code for each backend. With `BaseRetriever`:

```python
# This works with Chroma, FAISS, BM25, Wikipedia — anything
retriever.invoke("What is RAG?")
```

The chain doesn't care what's underneath. It just calls `.invoke()`.

### The Core Method

Every class that extends `BaseRetriever` must implement one method:

```python
def _get_relevant_documents(self, query: str) -> List[Document]:
    ...
```

This is the internal implementation. You should **never call it directly**.

```python
# ❌ Never do this
retriever._get_relevant_documents("my query")

# ✅ Always use the public interface
retriever.invoke("my query")
```

### Why `.invoke()` Instead of Calling `_get_relevant_documents` Directly?

`.invoke()` is inherited from `Runnable` and does more than just call the private method. It:

- Handles errors and callbacks
- Supports tracing and observability (LangSmith)
- Enables `.batch()` and `.stream()` variants
- Makes the retriever composable in LCEL chains

---

## 7. Building a Custom Retriever

### Theory

Since `BaseRetriever` just requires you to implement `_get_relevant_documents`, you can build a retriever backed by **anything** — a keyword search, an API, a database query, or a combination.

### Minimal Example: Keyword Retriever

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class SimpleKeywordRetriever(BaseRetriever):
    docs: List[Document]  # Pydantic field (required in modern LangChain)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Return docs that contain the query string (case-insensitive)."""
        query_lower = query.lower()
        return [
            doc for doc in self.docs
            if query_lower in doc.page_content.lower()
        ]
```

### Using It

```python
sample_docs = [
    Document(page_content="RAG improves LLM responses by retrieving context"),
    Document(page_content="LangChain is a framework for LLM applications"),
    Document(page_content="Chroma is a vector database for embeddings"),
]

retriever = SimpleKeywordRetriever(docs=sample_docs)

results = retriever.invoke("RAG")
for doc in results:
    print(doc.page_content)
# Output: "RAG improves LLM responses by retrieving context"
```

### This Custom Retriever Is Now a Full Runnable

```python
# Pipe it directly into an LCEL chain
chain = retriever | format_docs | prompt | llm | StrOutputParser()
chain.invoke("What is RAG?")
```

### Real-World Custom Retriever Use Cases

| Use Case                    | What `_get_relevant_documents` does                            |
| --------------------------- | -------------------------------------------------------------- |
| Hybrid search               | Combines vector similarity + BM25 keyword scores               |
| Metadata-filtered retrieval | Filters by date, category, author before returning             |
| API-backed retrieval        | Calls Wikipedia, Arxiv, or a REST API                          |
| Reranking                   | Retrieves 20 docs, reranks with a cross-encoder, returns top 5 |
| Database retrieval          | Queries SQL/NoSQL with the query string                        |

---

## 8. How Everything Connects

### The Retrieval Hierarchy

```
Vector Store (Chroma, FAISS, Pinecone, ...)
        │
        │  .as_retriever()
        ▼
  BaseRetriever  ◄──── Custom retrievers also inherit from here
        │
        │  .invoke(query)
        │        │
        │        └──► internally calls _get_relevant_documents(query)
        ▼
  List[Document]
        │
        │  format_docs()
        ▼
  Context string
        │
        ▼
  Prompt Template
        │
        ▼
       LLM
        │
        ▼
  Final Answer
```

### Where Each Concept Fits

| Concept                      | Role                                      | LCEL Compatible |
| ---------------------------- | ----------------------------------------- | --------------- |
| `similarity_search`          | Direct DB query, manual use               | ❌              |
| `as_retriever()`             | Wraps vector store as `BaseRetriever`     | ✅              |
| `BaseRetriever`              | Standard interface all retrievers follow  | ✅              |
| Custom Retriever             | Extends `BaseRetriever` with custom logic | ✅              |
| `RetrievalQA`                | Legacy prebuilt chain (retriever + LLM)   | Partially       |
| `RunnableWithMessageHistory` | Adds session memory to any LCEL chain     | ✅              |

---

## 9. Best Practices & Key Takeaways

### Retrieval

- ✅ Always use `as_retriever()` for LCEL pipelines — never use `similarity_search` in chains
- ✅ Use `similarity_search` only for debugging and testing what's in your vector store
- ✅ Avoid `RetrievalQA` in new projects — it's a legacy abstraction with limited flexibility
- ✅ Use `search_type="mmr"` when retrieved chunks tend to be repetitive or redundant
- ✅ Keep retrieval fast — it's often the **bottleneck** in RAG latency

### Custom Retrievers

- ✅ Extend `BaseRetriever` when you need logic that a vector store can't provide alone
- ✅ Always return `List[Document]` — never raw strings or dicts
- ✅ Always call `.invoke()` — never `._get_relevant_documents()` directly
- ✅ Use Pydantic fields (not `__init__`) for attributes in modern LangChain retrievers

### Embeddings

- ✅ Add retry logic around embedding calls — especially for hosted APIs
- ✅ Use exponential backoff in production to avoid hammering a rate-limited endpoint
- ✅ Persist your vector store — embedding thousands of chunks on every run is wasteful
- ✅ `embed_documents` and `embed_query` are different — both must be handled in wrappers

### General

- ✅ Return `List[Document]` from all retrievers — the rest of the chain depends on it
- ✅ Always include `source` metadata in documents — enables answer attribution
- ✅ Use `k=3` to `k=5` as a starting point; tune based on answer quality

---

## 10. Complete RAG Mental Model

```
[User Query]
     │
     ▼
[BaseRetriever]  ◄── Chroma.as_retriever() or Custom Retriever
     │                    └── internally: similarity search on vector store
     ▼
[List[Document]]  (top-k relevant chunks with metadata)
     │
     ▼
[format_docs()]  →  single context string with SOURCE labels
     │
     ▼
[ChatPromptTemplate]  →  system + chat_history + context + user query
     │
     ▼
[LLM (ChatGroq)]
     │
     ▼
[StrOutputParser]  →  clean response string
     │
     ▼
[RunnableWithMessageHistory]  →  saves to ChatMessageHistory for next turn
     │
     ▼
[Response to User]
```

---

## Quick Reference

| Class / Method                   | Module                      | One-Line Purpose                                 |
| -------------------------------- | --------------------------- | ------------------------------------------------ |
| `db.similarity_search(query, k)` | vector store                | Direct retrieval, returns `List[Document]`       |
| `db.as_retriever(search_kwargs)` | vector store                | Wraps store as a `BaseRetriever` Runnable        |
| `BaseRetriever`                  | `langchain_core.retrievers` | Abstract base class all retrievers implement     |
| `_get_relevant_documents(query)` | `BaseRetriever`             | Internal method to override in custom retrievers |
| `retriever.invoke(query)`        | `Runnable`                  | Public interface — always use this               |
| `RetrievalQA.from_chain_type()`  | `langchain.chains`          | Legacy prebuilt retriever + LLM chain            |
| `SafeEmbeddings`                 | custom                      | Retry wrapper for robust embedding calls         |

---

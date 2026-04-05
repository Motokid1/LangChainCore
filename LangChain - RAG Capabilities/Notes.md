---
# LangChain Core (LCEL) RAG Pipeline - Complete Guide

## Table of Contents

1. [Environment & LLM Setup](#environment--llm-setup)
2. [Document Loading](#document-loading)
3. [Text Splitting](#text-splitting)
4. [Embeddings](#embeddings)
5. [Vector Store & Persistent Storage](#vector-store--persistent-storage)
6. [Prompt Templates](#prompt-templates)
7. [Formatting Retrieved Documents](#formatting-retrieved-documents)
8. [LCEL RAG Chain](#lcel-rag-chain)
9. [Memory](#memory)
10. [Running the RAG Chat](#running-the-rag-chat)
11. [Best Practices & Key Takeaways](#best-practices--key-takeaways)
---

## 1. Environment & LLM Setup

**Theory:**
Before using LangChain, we need to securely load API keys and initialize the LLM (Groq in our case). Using `.env` prevents exposing credentials in code.

**Code:**

```python
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-120b")  # Initialize Groq LLM
```

**Key Points:**

- `.env` stores API keys like `OPENAI_API_KEY`.
- `load_dotenv()` loads these keys.
- `ChatGroq()` initializes the LLM for downstream tasks.

---

## 2. Document Loading

**Theory:**
Load documents from directories, handling TXT and PDF formats. This prepares the raw text for RAG.

**Code:**

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader

docs = []

# Load TXT
txt_loader = DirectoryLoader("data/", glob="**/*.txt", loader_cls=TextLoader)
docs.extend(txt_loader.load())

# Load PDF
pdf_loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs.extend(pdf_loader.load())

print(f"Loaded documents: {len(docs)}")
```

**Key Points:**

- `DirectoryLoader` iterates directories.
- `TextLoader` and `PyPDFLoader` read specific formats.
- Always validate if docs are loaded.

---

## 3. Text Splitting

**Theory:**
Large documents are split into chunks for embeddings. Overlap ensures context continuity.

**Code:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)
split_docs = [doc for doc in split_docs if doc.page_content.strip()]
```

**Key Points:**

- `chunk_size` controls token length per chunk.
- `chunk_overlap` maintains context across chunks.
- Removing empty chunks avoids errors downstream.

---

## 4. Embeddings

**Theory:**
Text chunks are converted to vector representations for semantic search.

**Code:**

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

**Key Points:**

- Embeddings enable semantic similarity retrieval.
- `HuggingFaceEmbeddings` is free and effective for small projects.

---

## 5. Vector Store & Persistent Storage

**Theory:**
Store embeddings in Chroma for persistent retrieval, enabling RAG to fetch relevant context efficiently.

**Code:**

```python
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
retriever = db.as_retriever(search_kwargs={"k": 3})
```

**Key Points:**

- `persist_directory` allows storage across sessions.
- `retriever` fetches top-k relevant chunks.

---

## 6. Prompt Templates

**Theory:**
Prompt templates structure LLM input. Multi-turn conversation and context placeholders help in coherent responses.

**Code:**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use the retrieved context to answer the question.
If answer not found, say "I don't know".
Include source references at the end.

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),  # Maintain chat context
    ("human", "{input}")
])
```

**Key Points:**

- Placeholders: `{input}`, `{context}`, `{chat_history}`.
- `MessagesPlaceholder` allows multi-turn conversation memory.

---

## 7. Formatting Retrieved Documents

**Theory:**
Convert retrieved chunks into a formatted string with source metadata for LLM.

**Code:**

```python
def format_docs(docs):
    return "\n\n".join(
        f"{doc.page_content}\nSOURCE: {doc.metadata.get('source', 'unknown')}"
        for doc in docs
    )
```

**Key Points:**

- Ensures source attribution in answers.
- Prepares context for the RAG chain.

---

## 8. LCEL RAG Chain

**Theory:**
Combine retriever, prompt, LLM, and output parser using LCEL pipes.

**Code:**

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def get_input_str(x):
    return x["input"] if isinstance(x, dict) else x

rag_chain = (
    {
        "context": RunnableLambda(get_input_str) | retriever | format_docs,
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", [])
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)
```

**Key Points:**

- LCEL allows chaining functions like a pipeline.
- `RunnableLambda` can wrap functions for flexibility.
- Output parser ensures clean string response.

---

## 9. Memory

**Theory:**
Store multi-turn conversations per user session using `RunnableWithMessageHistory`.

**Code:**

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

rag_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
```

**Key Points:**

- Session ID maintains separate chat memory per user.
- `ChatMessageHistory` stores previous messages for context.

---

## 10. Running the RAG Chat

**Theory:**
Interactive chat loop that retrieves context, runs RAG, and displays answers with memory.

**Code:**

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

**Key Points:**

- Interactive loop for real-time chat.
- Exit mechanism ensures graceful termination.

---

## 11. Best Practices & Key Takeaways

- **API Security:** Use `.env` for keys; never hardcode credentials.
- **Document Validation:** Always check loaded docs and remove empty chunks.
- **Chunking:** Overlap ensures better context for retrieval.
- **Source Attribution:** Format docs with sources for transparency.
- **Persistent Storage:** Chroma DB ensures embeddings are saved across sessions.
- **Memory Management:** Maintain session-based chat history to improve multi-turn conversation.
- **Error Handling:** Wrap loaders and chunkers in try/except to avoid pipeline breaks.
- **LCEL Approach:** Pipe functions for clean, modular, maintainable RAG pipelines.

---

✅ **Conclusion:**
With these steps, you can build a **complete RAG pipeline using LangChain Core (LCEL)** with **Groq LLM**, **persistent embeddings**, **source attribution**, and **memory-aware multi-turn conversation**.

---

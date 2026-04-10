# 🤖 Model Context Protocol (MCP) — A Complete Deep Dive

> **Your go-to reference for understanding MCP, its architecture, real-world usage, and how it compares to RAG.**

---

## Table of Contents

1. [What is MCP?](#1-what-is-mcp)
2. [Architecture: Host, Client, and Server](#2-architecture-host-client-and-server)
3. [MCP Servers — What Do They Expose?](#3-mcp-servers--what-do-they-expose)
4. [MCP vs. RAG — The Key Distinction](#4-mcp-vs-rag--the-key-distinction)
5. [The Ecosystem Shift — Goodbye, Walled Gardens](#5-the-ecosystem-shift--goodbye-walled-gardens)
6. [Minimal Code Examples](#6-minimal-code-examples)
7. [Bonus: When to Use What](#7-bonus-when-to-use-what)
8. [Glossary](#8-glossary)

---

## 1. What is MCP?

### Definition

**Model Context Protocol (MCP)** is an open, standardized protocol that allows AI models (like Claude, GPT, etc.) to **connect to external tools, data sources, and services** in a consistent, pluggable way.

- **Created by:** Anthropic (announced November 2024)
- **Type:** Open protocol (open-source SDKs, open specification)
- **Website:** [modelcontextprotocol.io](https://modelcontextprotocol.io)

### The Problem It Solves

Before MCP, connecting an AI to an external tool (say, your company's database or GitHub) required:

- Writing a **custom integration** for every single tool.
- Every AI vendor doing this differently — incompatible with each other.
- Developers rebuilding the same connectors over and over.

Think of it like the early days of computer peripherals. Every printer had its own driver, every mouse needed custom software. Then **USB** came along — one standard port, everything just works.

> **MCP is the USB port for AI tools.**

---

## 2. Architecture: Host, Client, and Server

MCP has three distinct roles that work together. Here's how they relate:

```
┌─────────────────────────────────────────────┐
│                 MCP HOST                     │
│  (Claude Desktop, VS Code, your custom app) │
│                                             │
│   ┌──────────────────────────────────────┐  │
│   │           MCP CLIENT                 │  │
│   │  (Lives inside the Host;             │  │
│   │   manages connections)               │  │
│   └──────────────┬───────────────────────┘  │
└──────────────────│──────────────────────────┘
                   │  JSON-RPC over stdio/SSE
         ┌─────────▼──────────┐
         │    MCP SERVER       │
         │ (A separate process │
         │  exposing tools)    │
         └────────────────────┘
              │          │
        ┌─────▼──┐  ┌────▼──────┐
        │ GitHub │  │ Local DB  │
        └────────┘  └───────────┘
```

### The Three Roles Explained

| Role | What It Is | Real-World Analogy |
|------|-----------|-------------------|
| **Host** | The application that runs the AI model. The user interacts with it. | The **browser** (Chrome, Firefox) |
| **Client** | A component *inside* the Host that speaks MCP. Manages one connection per server. | The **browser's network stack** |
| **Server** | A separate lightweight program that wraps a tool/data source and exposes it via MCP. | A **web server** serving an API |

### How They Communicate

MCP uses **JSON-RPC 2.0** as its message format. Communication happens over:

- **stdio** — for local servers (the Host spawns the server as a subprocess)
- **SSE (Server-Sent Events)** — for remote/networked servers

The lifecycle:
1. Host starts → spawns or connects to MCP Servers
2. Client sends `initialize` handshake → Server replies with its capabilities
3. AI decides it needs a tool → Client sends `tools/call` → Server executes → returns result
4. Result is injected back into the AI's context

---

## 3. MCP Servers — What Do They Expose?

An MCP Server can expose three types of primitives:

| Primitive | Description | Example |
|-----------|-------------|---------|
| **Tools** | Functions the AI can *call* (actions) | `search_files()`, `create_issue()` |
| **Resources** | Data the AI can *read* (like files or DB rows) | A file's contents, a database record |
| **Prompts** | Pre-defined prompt templates the user can invoke | A structured "code review" prompt |

### 3 Practical MCP Server Examples

---

#### Example 1: GitHub MCP Server

The official GitHub MCP Server exposes tools like:

- `list_repositories` — list repos for a user/org
- `create_issue` — open a new GitHub issue
- `get_pull_request` — fetch PR details
- `search_code` — search across repositories

**What this enables:** You can ask Claude: *"Open a GitHub issue titled 'Fix login bug' in my repo"* — and it actually does it.

---

#### Example 2: Google Drive MCP Server

Exposes resources and tools such as:

- `list_files` — browse Drive folders
- `read_file` — get the text content of a doc/sheet
- `search_files` — find files by name or content
- `create_document` — make a new Google Doc

**What this enables:** *"Summarize the Q3 report in my Drive"* — Claude reads the live file, not a stale copy.

---

#### Example 3: Local Database (SQLite/PostgreSQL) MCP Server

Exposes:

- `run_query` — execute a SELECT query
- `list_tables` — describe the database schema
- `insert_row` — write data (if permitted)

**What this enables:** *"How many users signed up last week?"* — Claude queries your actual production database (read-only mode recommended!).

---

## 4. MCP vs. RAG — The Key Distinction

This is where most people get confused. They solve **different problems**.

### Quick Analogy

| | Analogy |
|--|---------|
| **RAG** | A student studying flashcards before an exam. Knowledge is embedded at index time. |
| **MCP** | A student using a live calculator and terminal *during* the exam. Actions happen in real time. |

### Comparison Table

| Feature | RAG | MCP |
|---------|-----|-----|
| **Primary Goal** | Give AI long-term memory / domain knowledge | Give AI active tools and live data access |
| **Data State** | Usually static (indexed in a vector DB) | Live / Dynamic (real-time API or file state) |
| **Access Type** | Read-only (finding information) | Read **and** Write (can edit files, trigger actions) |
| **Latency** | Low (local vector search) | Variable (depends on external API/tool) |
| **Staleness Risk** | High (index can become outdated) | None (always fetches current state) |
| **Best For** | Company wikis, docs, FAQ, knowledge bases | GitHub, databases, calendars, live APIs |
| **Setup Complexity** | Medium (chunking, embedding, vector DB) | Low (run an MCP server, plug it in) |
| **Persistence** | Yes (embeddings stored long-term) | No (tool results are ephemeral in context) |

### Do They Replace Each Other?

**No. They are complementary.**

- Use **RAG** when you need the AI to know *about* a large corpus of documents (your internal wiki, product documentation).
- Use **MCP** when you need the AI to *do something* or fetch *real-time* data (check a live ticket, write to a file, query current metrics).

A production system might use **both**:
- RAG to answer *"What is our refund policy?"* (static knowledge)
- MCP to answer *"What is the status of order #12345 right now?"* (live data)

---

## 5. The Ecosystem Shift — Goodbye, Walled Gardens

### The "Walled Garden" Problem

Before MCP, each AI had proprietary plugin systems:

- OpenAI had **ChatGPT Plugins** (custom spec, deprecated)
- Google had its own tool-use format
- Anthropic had custom tool definitions
- Each IDE, each app — a different integration pattern

Result: **N tools × M AI systems = N×M custom integrations**. An enormous maintenance burden.

### MCP's Answer: N + M

With a standard protocol:

- Tool builders write **one MCP Server**.
- AI apps implement **one MCP Client**.
- Any server works with any client. **N + M integrations instead of N × M.**

### Why Standardization Wins

1. **Portability** — An MCP server built for Claude Desktop works with VS Code Copilot, Cursor, or your custom app, without changes.

2. **Security boundary** — The server is a separate process. It defines exactly what the AI can and cannot do. No arbitrary code execution.

3. **Community ecosystem** — Hundreds of community-built MCP servers already exist (Slack, Notion, Jira, Postgres, filesystem, web search...).

4. **Faster iteration** — Adding a new tool to your AI workflow goes from *weeks of integration work* to *adding one line to your config*.

---

## 6. Minimal Code Examples

### 6.1 — Writing a Minimal MCP Server (Python)

```python
# minimal_server.py
# Install: pip install mcp

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio

# Create the server
app = Server("my-first-mcp-server")

# Register a tool
@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="say_hello",
            description="Says hello to a given name",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name to greet"}
                },
                "required": ["name"]
            }
        )
    ]

# Handle tool calls
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "say_hello":
        return [TextContent(type="text", text=f"Hello, {arguments['name']}!")]

# Run the server over stdio
async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

asyncio.run(main())
```

---

### 6.2 — MCP Server Config (Claude Desktop)

```json
// ~/.config/claude/claude_desktop_config.json
// This tells Claude Desktop to spawn your server

{
  "mcpServers": {
    "my-first-server": {
      "command": "python",
      "args": ["/path/to/minimal_server.py"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/you/Documents"]
    }
  }
}
```

---

### 6.3 — Using the MCP Python Client (Programmatic)

```python
# client_example.py
# Connect to an MCP server programmatically

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Point to your server script
    server_params = StdioServerParameters(
        command="python",
        args=["minimal_server.py"]
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print("Available tools:", [t.name for t in tools.tools])

            # Call a tool
            result = await session.call_tool("say_hello", {"name": "World"})
            print("Result:", result.content[0].text)
            # Output: Hello, World!

asyncio.run(main())
```

---

### 6.4 — A More Useful Server: Wrapping a Real API

```python
# weather_server.py
# An MCP server that wraps a weather API

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import asyncio, json

app = Server("weather-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        city = arguments["city"]
        # Using Open-Meteo (free, no API key needed for demo)
        url = f"https://wttr.in/{city}?format=j1"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            data = resp.json()
            temp = data["current_condition"][0]["temp_C"]
            desc = data["current_condition"][0]["weatherDesc"][0]["value"]
            return [TextContent(type="text", text=f"{city}: {temp}°C, {desc}")]

async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write, app.create_initialization_options())

asyncio.run(main())
```

---

### 6.5 — A Minimal RAG Pipeline (for Comparison)

```python
# minimal_rag.py
# Simple RAG: embed docs → store → query
# Install: pip install sentence-transformers numpy

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Your "knowledge base" (static documents)
documents = [
    "Our refund policy allows returns within 30 days.",
    "The office is open Monday to Friday, 9am to 5pm.",
    "Our API rate limit is 1000 requests per hour.",
]

# Step 2: Embed and store (this is your "index")
doc_embeddings = model.encode(documents)

def retrieve(query: str, top_k: int = 1) -> list[str]:
    """Find the most relevant document for a query."""
    query_embedding = model.encode([query])
    
    # Cosine similarity
    scores = np.dot(doc_embeddings, query_embedding.T).flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    return [documents[i] for i in top_indices]

# Step 3: Query
query = "Can I return a product?"
results = retrieve(query)
print(f"Query: {query}")
print(f"Retrieved: {results[0]}")
# Output: "Our refund policy allows returns within 30 days."

# In a real RAG system, you'd then pass this to an LLM along with the query.
# MCP would instead call a live "get_policy" tool in real-time.
```

---

## 7. Bonus: When to Use What

```
                    Do you need LIVE data or ACTIONS?
                           /             \
                         YES              NO
                          |               |
                  Use MCP Tools      Is it a large document
                  (GitHub, DB,       corpus you need to
                  APIs, files)       search semantically?
                                         /        \
                                       YES          NO
                                        |            |
                                   Use RAG       Just include
                                  (vector DB,    it in the
                                  embeddings)    system prompt
```

### Decision Cheat Sheet

| Scenario | Solution |
|----------|----------|
| "Summarize our 500-page internal wiki" | RAG |
| "What's the current status of ticket #42?" | MCP |
| "Answer questions about our product docs" | RAG |
| "Create a GitHub issue from this bug report" | MCP |
| "What did our CEO say in the last all-hands?" | RAG (if transcript indexed) |
| "Send this summary to Slack #engineering" | MCP |
| "How many orders came in today?" | MCP (live DB query) |

---

## 8. Glossary

| Term | Meaning |
|------|---------|
| **MCP** | Model Context Protocol — open standard for AI-tool connectivity |
| **Host** | The app running the AI (Claude Desktop, VS Code, etc.) |
| **Client** | MCP component inside the Host; manages server connections |
| **Server** | Separate process exposing tools/resources via MCP |
| **Tool** | A callable function exposed by a server (an action) |
| **Resource** | A readable data object exposed by a server |
| **RAG** | Retrieval-Augmented Generation — augmenting AI with a searchable knowledge base |
| **Vector DB** | Database storing embeddings for semantic search (used in RAG) |
| **JSON-RPC** | The message format MCP uses for Host↔Server communication |
| **stdio** | Standard input/output — the local transport method for MCP |
| **SSE** | Server-Sent Events — the remote/networked transport method for MCP |
| **Embedding** | A numeric vector representation of text used for semantic similarity |

---

## Further Reading

- 📖 [Official MCP Specification](https://spec.modelcontextprotocol.io)
- 🛠️ [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- 🛠️ [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- 🗂️ [Community MCP Servers](https://github.com/modelcontextprotocol/servers)
- 📺 [Anthropic MCP Introduction Blog Post](https://www.anthropic.com/news/model-context-protocol)

---

*Made with ❤️ for anyone trying to understand MCP clearly.*

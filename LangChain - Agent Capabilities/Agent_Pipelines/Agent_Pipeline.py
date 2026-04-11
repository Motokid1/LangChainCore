#Architectural Pattern: Agent Pipeline with RAG and Tools
#                 ┌─────────────────────────┐
# User Query ───▶│        AGENT            │
#                 │ (create_agent)         │
#                 └─────────┬──────────────┘
#                           │
#         ┌─────────────────┼──────────────────┐
#         │                 │                  │
#    add tool         rag_search tool     summarize tool
#         │                 │                  │
#         │           ┌─────────────┐          │
#         │           │   LCEL RAG  │          │
#         │           │ pipeline    │          │
#         │           └─────────────┘          │
#         │                                    │
#    math result                        summary result

# 1. Imports
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

# 2. Constants
DATA_PATH = r"D:\Gen AI\data"
CHROMA_PATH = r"D:\Gen AI\chroma_db"

# 3. Initialize LLM
llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0
)

parser = StrOutputParser()

# 4. Load Documents
def load_all_docs():
    pdf_docs = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    ).load()

    text_docs = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader
    ).load()

    return pdf_docs + text_docs

# 5. Split Documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = load_all_docs()
chunks = splitter.split_documents(documents)

# 6. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 7. Chroma Vector Store (PERSISTENT)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)

# Save to disk
vectorstore.persist()

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 8. RAG Chain (LCEL)
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    RunnableLambda(lambda x: x["query"])
    | RunnableLambda(lambda q: {"docs": retriever.invoke(q), "query": q})
    | RunnableLambda(lambda x: {
        "context": format_docs(x["docs"]),
        "query": x["query"]
    })
    | RunnableLambda(
        lambda x: f"""
        Answer based on the context.

        Context:
        {x['context']}

        Question:
        {x['query']}
        """
    )
    | llm
    | parser
)

# 9. Tools
@tool
def rag_search(query: str) -> str:
    """Search and answer using stored documents (Chroma RAG)"""
    return rag_chain.invoke({"query": query})

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@tool
def power(a: int, b: int) -> int:
    """Raise a to the power of b"""
    return a ** b

@tool
def divide(a: int, b: int) -> float:
    """Divide a by b"""
    return a / b if b != 0 else 0

# 10. Summarizer
summarizer_chain = (
    RunnableLambda(lambda text: f"Summarize:\n{text}")
    | llm
    | parser
)

@tool
def summarize_text(text: str) -> str:
    """Summarize given text"""
    return summarizer_chain.invoke(text)

# 11. Create Agent
agent = create_agent(
    model=llm,
    tools=[
        rag_search,
        add,
        multiply,
        power,
        divide,
        summarize_text
    ],
    system_prompt="""
    You are an advanced AI agent.

    Capabilities:
    - Answer from documents using Chroma RAG
    - Perform math operations
    - Summarize content

    Rules:
    - Use RAG for document-based questions
    - Use math tools when needed
    - Think step-by-step
    """
)

# 12. Input Builder
def build_input(user_input: str):
    return {
        "messages": [
            SystemMessage(content="You are an intelligent assistant"),
            HumanMessage(content=user_input)
        ]
    }

input_chain = RunnableLambda(build_input)

# 13. Agent Pipeline
agent_chain = input_chain | agent

# 14. Extract Final Answer
def get_final_answer(response: dict):
    return response["messages"][-1].content

# 15. Run
if __name__ == "__main__":
    while True:
        query = input("\nEnter query (or 'exit'): ")

        if query.lower() == "exit":
            break

        response = agent_chain.invoke(query)

        print("\n===== TOOL CALLS =====\n")
        print(get_final_answer(response))
        for msg in response["messages"]:
            if getattr(msg, "tool_calls", None):
                print(msg.tool_calls)

        print("\n===== FINAL RESPONSE =====\n")
        
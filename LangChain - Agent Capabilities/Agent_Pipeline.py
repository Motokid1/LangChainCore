# 1. Imports
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader
)

# 2. Constants
DATA_PATH = r"D:\Gen AI\data"

# 3. Initialize LLM
llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0
)

# 4. Helper Function
def load_docs(loader):
    docs = loader.load()
    return "\n".join([doc.page_content for doc in docs])[:3000]

# 5. LCEL Loader Chains
pdf_loader_chain = RunnableLambda(
    lambda _: load_docs(
        DirectoryLoader(
            DATA_PATH,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
    )
)

text_loader_chain = RunnableLambda(
    lambda _: load_docs(
        DirectoryLoader(
            DATA_PATH,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
    )
)

# 6. Tools 
@tool
def load_pdfs(_: str = "") -> str:
    """Load all PDF documents from the data folder"""
    return pdf_loader_chain.invoke(None)
#what does the _ do in the function signature?
# The underscore (_) in the function signature is a common convention in Python to indicate that the parameter is intentionally unused. 
# It serves as a placeholder to show that the function expects an argument, but that argument will not be utilized within the function body. 
# In this case, the load_pdfs and load_texts functions are designed to be called without any meaningful input, so the underscore is used to signify that the input parameter is not relevant to the function's operation.

@tool
def load_texts(_: str = "") -> str:
    """Load all text files from the data folder"""
    return text_loader_chain.invoke(None)

# 7. Math Tools (WITH DOCSTRINGS ✅)
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
    """Divide a by b (handles division by zero)"""
    return a / b if b != 0 else 0

# 8. Summarizer (LCEL)
summarizer_chain = RunnableLambda(
    lambda text: f"Summarize this:\n{text}"
) | llm

@tool
def summarize_text(text: str) -> str:
    """Summarize the given text content"""
    return summarizer_chain.invoke(text).content

# 9. Create Agent
agent = create_agent(
    model=llm,
    tools=[
        load_pdfs,
        load_texts,
        add,
        multiply,
        power,
        divide,
        summarize_text
    ],
    system_prompt="""
    You are an advanced AI agent.

    Capabilities:
    - Load PDFs and text files
    - Perform math operations
    - Summarize content

    Rules:
    - Use tools when needed
    - Think step-by-step
    - Give clear final answers
    """
)

# 10. Input Builder (CRITICAL)
def build_input(user_input: str):
    return {
        "messages": [
            SystemMessage(content="You are an intelligent assistant"),
            HumanMessage(content=user_input)
        ]
    }

input_chain = RunnableLambda(build_input)

# 11. LCEL Agent Pipeline
agent_chain = input_chain | agent

# 12. Extract Final Answer (FIX)
def get_final_answer(response: dict):
    return response["messages"][-1].content

# 13. Run
if __name__ == "__main__":
    while True:
        query = input("\nEnter your query (or 'exit'): ")

        if query.lower() == "exit":
            break

        response = agent_chain.invoke(query)

        print("\n===== FINAL RESPONSE =====\n")
        print(get_final_answer(response))
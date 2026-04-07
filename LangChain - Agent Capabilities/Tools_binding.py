import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

# Define tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Bind tool
llm_with_tools = llm.bind_tools([add_numbers])

response = llm_with_tools.invoke("What is 5 + 3?")

print("\nAI Response:\n", response.content)
print("\nTool Calls:\n", response.tool_calls)
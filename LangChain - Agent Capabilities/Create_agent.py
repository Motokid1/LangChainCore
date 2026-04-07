import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

# Initialize model
llm = ChatGroq(model="openai/gpt-oss-120b")

# Define tool
@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

# Create agent (NEW API)
agent = create_agent(
    model=llm,
    tools=[add],
    system_prompt="You are a helpful math assistant"
)

# Invoke agent
response = agent.invoke({
    "messages": [
        ("user", "What is 15 + 25?")
    ]
})

print("\nAgent Output:\n", response)
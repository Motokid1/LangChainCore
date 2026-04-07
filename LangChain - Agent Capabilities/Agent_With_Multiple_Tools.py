import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

# Initialize model
llm = ChatGroq(model="openai/gpt-oss-120b")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply numbers"""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

agent = create_agent(
    model=llm,
    tools=[add, multiply],
    system_prompt="You are a calculator assistant"
)

response = agent.invoke({
    "messages": [
        ("user", "What is (5 + 3) * 2?")
    ]
})

print(response)
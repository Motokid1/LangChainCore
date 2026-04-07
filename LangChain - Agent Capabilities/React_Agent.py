import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply numbers"""
    return a * b

# create_react_agent handles the loop: 
# 1. Ask LLM -> 2. Run Tool -> 3. Give result back to LLM -> 4. Final Answer
agent = create_react_agent(
    model=llm,
    tools=[multiply]
)

# Invoke returns the full state dictionary
result = agent.invoke({
    "messages": [("user", "What is 7 * 6?")]
})

# The answer is the 'content' of the LAST message in the state
final_message = result["messages"][-1]

print("\n--- Full State Messages ---")
for msg in result["messages"]:
    msg.pretty_print()

print("\n--- Just the Answer ---")
print(final_message.content)
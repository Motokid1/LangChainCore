import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# Load env
load_dotenv()

# Initialize model
llm = ChatGroq(model="openai/gpt-oss-120b")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is LangChain?")
]

response = llm.invoke(messages)

print("\nAI Response:\n", response.content)

# Simulating tool message (normally comes from tool execution)
tool_msg = ToolMessage(
    content="Weather is 30°C",
    tool_call_id="call_1"
)

print("\nTool Message Example:\n", tool_msg)
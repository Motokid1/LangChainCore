import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq"
)

messages = [
    SystemMessage(content="You are a strict coding assistant. Give short answers."),
    HumanMessage(content="Explain Python decorators")
]

response = llm.invoke(messages)

print("\nResponse:\n", response.content)
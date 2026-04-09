import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables
load_dotenv()

# Initialize model (Groq)
llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq"
)

# Simple call
response = llm.invoke("What is LangChain?")

print("\nResponse:\n", response.content)
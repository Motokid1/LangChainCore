import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that extracts user details."),  
    ("user", "Rohith is 22 years old and lives in Hyderabad. He enjoys hiking and cooking.")
])

# Define schema
class User(BaseModel):
    name: str
    age: int
    city: str
    hobbies: list[str]
    LinkedIn: str | None = None

# Structured LLM
structured_llm = llm.with_structured_output(User)

chain = prompt | structured_llm

response = chain.invoke({})
print("\nStructured Output:\n", response)
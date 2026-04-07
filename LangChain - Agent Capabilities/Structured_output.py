import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

# Define schema
class Person(BaseModel):
    name: str
    age: int
    city: str

# Force structured output
structured_llm = llm.with_structured_output(Person)

response = structured_llm.invoke(
    "Extract details: Rohith is 22 years old and lives in Hyderabad"
)

print("\nStructured Output:\n", response)
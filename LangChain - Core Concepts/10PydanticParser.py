from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Define schema
class Person(BaseModel):
    name: str
    age: int

# Create parser
parser = PydanticOutputParser(pydantic_object=Person)

# LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# IMPORTANT: Strong JSON instruction
prompt = ChatPromptTemplate.from_messages([
    ("system", "You MUST return ONLY valid JSON. No explanation."),
    ("human", "Generate a random person in JSON with fields: name and age")
])

# Chain
chain = prompt | llm | parser

# Run
result = chain.invoke({}) 
# Understand why we pass empty dict → no variables in prompt

print(result)
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatGroq(model="openai/gpt-oss-120b")

parser = JsonOutputParser()

# Ask LLM to return JSON
prompt = ChatPromptTemplate.from_messages([
    ("system", "Return output in JSON format"),
    ("human", "Give name and age of a fictional person")
])

chain = prompt | llm | parser

result = chain.invoke({})

# Output will be Python dict
print(result.content)
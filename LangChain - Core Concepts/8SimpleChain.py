# Simple chain = prompt → llm → parser

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="openai/gpt-oss-120b")

prompt = PromptTemplate.from_template(
    "Explain {topic}"
)

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"topic": "Docker"}))
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="openai/gpt-oss-120b")

# ChatPromptTemplate uses roles
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI tutor"),  # system message
    ("human", "Explain {topic}")               # user message
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "LangChain"})

print(result)
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    #max_tokens=500
)
# Dynamic system message
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in {domain}"),
    ("human", "Explain {topic}")
])

chain = prompt | llm | StrOutputParser()

result = chain.invoke({
    "domain": "Cloud Computing",
    "topic": "AWS Lambda"
})

print(result)
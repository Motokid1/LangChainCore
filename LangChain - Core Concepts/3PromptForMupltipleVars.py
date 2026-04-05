from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="openai/gpt-oss-120b")

# Multiple variables
prompt = PromptTemplate.from_template(
    "Explain {topic} in {level} level with examples"
)

parser = StrOutputParser()

chain = prompt | llm | parser

# Pass multiple inputs
result = chain.invoke({
    "topic": "Microservices",
    "level": "2 lines"
})

print(result)
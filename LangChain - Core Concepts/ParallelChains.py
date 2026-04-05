from dotenv import load_dotenv
load_dotenv()

from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="openai/gpt-oss-120b")

# Two parallel tasks
chain = RunnableParallel({
    "definition": PromptTemplate.from_template("Define {topic}") | llm ,

    "example": PromptTemplate.from_template("Give example of {topic}") | llm 
})

result = chain.invoke({"topic": "REST API"})

# print(result.content)

print(result["definition"].content)
print(result["example"].content)
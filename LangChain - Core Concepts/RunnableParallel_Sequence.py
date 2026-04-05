from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

parser = StrOutputParser()
# Create sequences instead of using |
definition_chain = RunnableSequence(
    RunnableLambda(lambda x: f"Define {x['topic']} in 2 lines"),
    llm, 
    parser
)

example_chain = RunnableSequence(
    RunnableLambda(lambda x: f"Give an example of {x['topic']} in 2 lines"),
    llm,
    parser
)

# Parallel execution
parallel = RunnableParallel({
    "definition": definition_chain,
    "example": example_chain
})

result = parallel.invoke({"topic": "LangChain"})

print(result.content)
# print(result["definition"].content)
# print(result["example"].content)
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

parser = StrOutputParser()
# Define parallel tasks
parallel = RunnableParallel({
    "definition": RunnableLambda(lambda x: f"Define {x['topic']} simply") | llm | parser,
    "example": RunnableLambda(lambda x: f"Give an example of {x['topic']}") | llm | parser,
    "importance": RunnableLambda(lambda x: f"Why is {x['topic']} important?") | llm | parser
})

# Invoke
result = parallel.invoke({"topic": "APIs"})

# Output
print(result["definition"])
print(result["example"])
print(result["importance"])
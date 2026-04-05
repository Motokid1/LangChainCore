from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model="openai/gpt-oss-120b")

# First prompt
prompt = PromptTemplate.from_template(
    "Explain {topic} in simple terms"
)

# Custom Python function inside chain
def add_more(text):
    return text + "\n\nAdd more details."

# Multi-step chain
chain = (
    prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(add_more)  # modify output
    | llm
    | StrOutputParser()
)

print(chain.invoke({"topic": "API"}))
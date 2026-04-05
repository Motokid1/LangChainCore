from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-120b")

# Prompt
prompt = PromptTemplate.from_template(
    "Explain {topic} in simple terms"
)

# Output parser converts AIMessage → string
parser = StrOutputParser()

# Chain: Prompt → LLM → Parser
chain = prompt | llm | parser

# Now output is directly string
result = chain.invoke({"topic": "FastAPI"})

print(result)
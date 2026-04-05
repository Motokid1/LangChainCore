# Load environment variables (.env file)
from dotenv import load_dotenv
load_dotenv()

# Import Groq LLM
from langchain_groq import ChatGroq

# Import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import Runnable

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,  # deterministic output
    max_tokens=500
)

# Create prompt
prompt = PromptTemplate.from_template(
    "Explain {topic} in simple terms"
)

# Create chain using LCEL (pipe operator)
chain = prompt | llm

# Invoke chain with input
response = chain.invoke({"topic": "LangChain"})

# response is AIMessage object → access .content
print(response.content)



from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda

# Initialize LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=500
)

# Prompt
prompt = PromptTemplate.from_template(
    "Explain {topic} in 2 lines"
)

# Convert to runnable
prompt_runnable = RunnableLambda(
    lambda inputs: prompt.format(**inputs)
)

# ✅ Correct way
sequence = RunnableSequence(
    prompt_runnable,
    llm
)

# Invoke
response = sequence.invoke({"topic": "Rohith"})

print(response.content)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from datetime import datetime

# 1. A dummy function to simulate fetching data (e.g., from a database)
def get_current_time(_):
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 2. Define the prompt
prompt = ChatPromptTemplate.from_template("""
Question: {question}
Current Time: {current_time}

Answer the question based on the time provided.
""")

llm = ChatGroq(model="openai/gpt-oss-120b")

# 3. The Complicated Chain
# assign() allows us to add 'current_time' while 'question' passes through automatically
chain = (
    RunnablePassthrough.assign(current_time=get_current_time)
    | prompt 
    | llm
)

# When we invoke, 'question' goes into RunnablePassthrough.
# assign() creates 'current_time'.
# Both are then sent to the prompt.
result = chain.invoke({"question": "Is it currently morning or evening?"})

print(result.content)
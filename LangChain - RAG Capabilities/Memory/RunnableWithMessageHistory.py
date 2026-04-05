from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(model="openai/gpt-oss-120b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful"),
    ("human", "{input}")
])

chain = prompt | llm

store = {}

def get_session(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session,
    input_messages_key="input"
)

# Run
res1 = chain_with_memory.invoke(
    {"input": "What is AI?"},
    config={"configurable": {"session_id": "1"}}
)

res2 = chain_with_memory.invoke(
    {"input": "Explain more"},
    config={"configurable": {"session_id": "1"}}
)

print(res2.content)
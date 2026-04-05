from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatGroq(model="openai/gpt-oss-120b")

# Prompt with memory placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder(variable_name="history"),  # memory goes here
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

# Store memory per session
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        print(f"Created new session with id: {session_id}")
    return store[session_id]

# Add memory to chain
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# First interaction
print(chain_with_memory.invoke(
    {"input": "Hi, I am Rohith"},
    config={"configurable": {"session_id": "1"}}
))

# Second interaction (uses memory)
print(chain_with_memory.invoke(
    {"input": "What is my name?"},
    config={"configurable": {"session_id": "1"}}
))
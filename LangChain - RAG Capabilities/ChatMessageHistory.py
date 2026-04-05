from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.chat_history import InMemoryChatMessageHistory

llm = ChatGroq(model="openai/gpt-oss-120b")

history = InMemoryChatMessageHistory()

history.add_user_message("What is AI?")
response = llm.invoke(history.messages)

history.add_ai_message(response.content)

print(history.messages)
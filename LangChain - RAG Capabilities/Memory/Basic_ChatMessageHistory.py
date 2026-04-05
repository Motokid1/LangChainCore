from langchain_core.chat_history import InMemoryChatMessageHistory

history = InMemoryChatMessageHistory()

history.add_user_message("Hello")
history.add_ai_message("Hi!")

print(history.messages)
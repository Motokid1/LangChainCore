from dotenv import load_dotenv
load_dotenv()

from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

llm = ChatGroq(model="llama-3.1-8b-instant")

@tool
def add(a: int, b: int) -> int:
    return a + b

agent = create_react_agent(llm, tools=[add])

response = agent.invoke({
    "messages": [("user", "add 5 and 7")]
})

print(response)
# Architecture 
#                 ┌──────────────────────┐
#                 │      User Input      │
#                 └─────────┬────────────┘
#                           │
#                           ▼
#                 ┌──────────────────────┐
#                 │ {"input": "..."}     │
#                 └─────────┬────────────┘
#                           │
#                           ▼
#         ┌────────────────────────────────────┐
#         │ RunnableWithMessageHistory         │
#         │ (Manages session-based memory)     │
#         └─────────┬──────────────────────────┘
#                   │
#                   ▼
#         ┌────────────────────────────────────┐
#         │ Normalize Input                    │
#         │ (Handles str / dict / message)     │
#         └─────────┬──────────────────────────┘
#                   │
#                   ▼
#         ┌────────────────────────────────────┐
#         │ Router (Decision Layer)            │
#         │ if weather → Weather Tool          │
#         │ if location → Location Tool        │
#         │ else → LLM                         │
#         └───────┬───────────┬───────────────┘
#                 │           │
#                 │           │
#         ┌───────▼──────┐    │
#         │ Location Tool│    │
#         │ (IP API)     │    │
#         └───────┬──────┘    │
#                 │           │
#                 ▼           │
#         ┌───────────────┐   │
#         │ Weather Tool  │◄──┘
#         │ (OpenWeather) │
#         └───────┬───────┘
#                 │
#                 ▼
#         ┌──────────────────────────┐
#         │ LLM Chain (Fallback)     │
#         │ Prompt + History + LLM   │
#         └─────────┬────────────────┘
#                   │
#                   ▼
#         ┌──────────────────────────┐
#         │ {"output": "..."}        │
#         │ + metadata (source)      │
#         └─────────┬────────────────┘
#                   │
#                   ▼
#         ┌──────────────────────────┐
#         │ Memory Update            │
#         │ (stores input & output)  │
#         └─────────┬────────────────┘
#                   │
#                   ▼
#         ┌──────────────────────────┐
#         │ Final Response to User   │
#         └──────────────────────────┘

# 1. Imports
import os
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Load ENV
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found")

# 2. LLM
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# 3. TOOLS
@tool
def get_current_location() -> str:
    """Get current user location based on IP"""
    res = requests.get("http://ip-api.com/json/", timeout=5).json()
    if res.get("status") != "success":
        return "Unable to fetch location"
    return f"{res.get('city')}, {res.get('country')}"

@tool
def get_weather(location: str) -> str:
    """Get weather for a given location"""
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    res = requests.get(url, timeout=5).json()

    if res.get("cod") != 200:
        return f"Weather error: {res.get('message')}"

    return f"{location}: {res['main']['temp']}°C, {res['weather'][0]['description']}"


tools = [get_current_location, get_weather]

# 4. MEMORY (simple store)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# 5. SYSTEM PROMPT
system_prompt = """
You are a helpful assistant.

Rules:
- Use tools when required
- If user asks weather → use get_weather
- If location needed → use get_current_location
- Be concise and accurate
"""

# 6. CREATE AGENT (NO ROUTER)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
) 

# 7. RUN LOOP WITH MEMORY
if __name__ == "__main__":
    print("🚀 Agent with Memory Started (create_agent)")

    session_id = "user1"

    while True:
        query = input("\nEnter query: ")

        if query.lower() == "exit":
            break

        # get history
        history = get_session_history(session_id).messages

        response = agent.invoke({
            "messages": history + [("user", query)]
        })

        print("\n===== RESPONSE =====")
        for msg in response["messages"]:
            if getattr(msg, "tool_calls", None):
                print(msg.tool_calls)
        print(response["messages"][-1].content)

# 1. LangChain (with `RunnableWithMessageHistory`) expects the response key to be `"output"`, not `"answer"`.

# 2. Since you returned `"answer"`, LangChain couldn’t store the AI response in memory, causing errors.

# 3. Only `"output"` is used to save conversation history; any other keys are ignored for memory.

# 4. By changing the router to return `{"output": ..., "source": ...}`, we ensure that the response is correctly stored in memory and can be retrieved in future interactions.

# Note:  
# The `get_input_str` function is crucial for ensuring that the retriever receives a clean string query, preventing type errors when the input is passed through the chain. 
# It checks if the input is a dictionary and extracts the "input" key's value. If it's not a dictionary, it returns the input as is. 
# This allows the RAG chain to function correctly without type errors, especially since the retriever expects a string query.

# Important Fixes in the Code:
# 1. Changed the router's return key from `"answer"` to `"output"` to ensure compatibility with `RunnableWithMessageHistory`.
# 2. Added an input normalization step to handle different input formats (str, dict, message objects) and extract the relevant query text for processing.

# Checkpoint: 
# "My current implementation uses LCEL with a manual router, 
# which works but becomes hard to scale due to conditional logic.

# LangGraph provides a more structured, state-driven approach 
# where control flow is explicitly defined using nodes and edges, 
# making it better suited for complex, production-grade agent systems."
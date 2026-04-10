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

from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 2. Load ENV
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    raise ValueError("❌ OPENWEATHER_API_KEY not found")

# 3. LLM
llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0
)

parser = StrOutputParser()

# 4. TOOLS
@tool
def get_current_location() -> str:
    """Get current user location"""
    try:
        res = requests.get("http://ip-api.com/json/", timeout=5).json()
        if res.get("status") != "success":
            return "Unable to fetch location"
        return f"{res.get('city')}, {res.get('country')}"
    except Exception as e:
        return f"Location error: {str(e)}"


@tool
def get_weather(location: str) -> str:
    """Get weather"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()

        if res.get("cod") != 200:
            return f"Weather error: {res.get('message')}"

        return f"{location}: {res['main']['temp']}°C, {res['weather'][0]['description']}"
    except Exception as e:
        return f"Weather error: {str(e)}"

# 5. MEMORY STORE
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# 6. PROMPT (WITH MEMORY)
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a controlled assistant.

Rules:
- Be concise
- Use conversation history if relevant
- Do not hallucinate
- Say "I don't know" if unsure
"""),
    ("placeholder", "{history}"),
    ("human", "{input}")
])

# 7. LLM CHAIN
llm_chain = prompt | llm | parser

# 8. INPUT NORMALIZATION
def extract_text(x):
    if isinstance(x, str):
        return x
    elif isinstance(x, dict):
        return x.get("input", "")
    elif hasattr(x, "content"):
        return x.content
    else:
        return str(x)

def normalize_input(x):
    return {
        "input": extract_text(x["input"]),
        "history": x.get("history", [])
    }

# 9. ROUTER (FIXED)
def router(x):
    query_text = x["input"]
    query = query_text.lower()

    # Weather
    if "weather" in query:
        loc = get_current_location.invoke({})
        weather = get_weather.invoke({"location": loc})
        return {"output": weather, "source": "weather"}

    # Location
    elif "location" in query:
        loc = get_current_location.invoke({})
        return {"output": loc, "source": "location"}

    # LLM
    else:
        answer = llm_chain.invoke({
            "input": query_text,
            "history": x.get("history", [])
        })
        return {"output": answer, "source": "llm"}

# 10. LCEL CHAIN
chain = (
    RunnableLambda(normalize_input)
    | RunnableLambda(router)
)

# 11. MEMORY WRAPPER
chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 12. RUN
if __name__ == "__main__":
    print("🚀 Agent with Memory Started")

    session_id = "user1"

    while True:
        query = input("\nEnter query: ")

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        try:
            response = chain_with_memory.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )

            print("\n===== RESPONSE =====")
            print("Answer :", response["output"])   # ✅ FIXED
            print("Source :", response["source"])

        except Exception as e:
            print("\n❌ Error:", str(e))

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
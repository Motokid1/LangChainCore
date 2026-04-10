# Architecture 
        #         ┌────────────────────────┐
        #         │      User Input        │
        #         └──────────┬─────────────┘
        #                    │
        #                    ▼
        # ┌──────────────────────────────────┐
        # │        Router / Controller       │
        # │ (if weather / location / LLM)    │
        # └──────────┬───────────┬──────────┘
        #            │           │
        # ┌──────────▼───┐   ┌──▼────────────┐
        # │ Location Tool │   │ Weather Tool │
        # │ (IP API)      │   │ (OpenWeather)│
        # └───────────────┘   └───────────────┘
        #            │
        #            └──────────┬──────────────┐
        #                       │              │
        #                       ▼              ▼
        #          ┌──────────────────────────────┐
        #          │        Memory Layer          │
        #          │  JSON File (memory_store)    │
        #          └──────────────┬───────────────┘
        #                         │
        #                         ▼
        #          ┌──────────────────────────────┐
        #          │ Memory Builder Function      │
        #          │ (build_memory_context)       │
        #          │ - reads JSON                 │
        #          │ - formats last N messages    │
        #          └──────────────┬───────────────┘
        #                         │
        #                         ▼
        #          ┌──────────────────────────────┐
        #          │ Prompt Construction Layer    │
        #          │ System + Memory + Input      │
        #          └──────────────┬───────────────┘
        #                         │
        #                         ▼
        #          ┌──────────────────────────────┐
        #          │     LLM (Groq / OpenAI)      │
        #          │  Reasoning + Response Gen    │
        #          └──────────────┬───────────────┘
        #                         │
        #                         ▼
        #          ┌──────────────────────────────┐
        #          │   Response to User           │
        #          └──────────────┬───────────────┘
        #                         │
        #                         ▼
        #          ┌──────────────────────────────┐
        #          │ Memory Storage Update        │
        #          │ Save Q/A → JSON File         │
        #          └──────────────────────────────┘
# 1. Imports
import os
import json
import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 2. ENV
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
    """Get current weather for a given location."""
    try:
        res = requests.get("http://ip-api.com/json/", timeout=5).json()
        if res.get("status") != "success":
            return "Unable to fetch location"
        return f"{res.get('city')}, {res.get('country')}"
    except Exception as e:
        return f"Location error: {str(e)}"


@tool
def get_weather(location: str) -> str:
    """Get current weather for a given location."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()

        if res.get("cod") != 200:
            return f"Weather error: {res.get('message')}"

        return f"{location}: {res['main']['temp']}°C, {res['weather'][0]['description']}"
    except Exception as e:
        return f"Weather error: {str(e)}"

# 5. MEMORY (JSON STORE)
MEMORY_FILE = "memory_store.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(store):
    with open(MEMORY_FILE, "w") as f:
        json.dump(store, f, indent=2)

store = load_memory()

# 6. BUILD MEMORY CONTEXT (KEY FIX)
def build_memory_context(session_id):
    if session_id not in store:
        return ""

    history = store[session_id]

    context = "Previous conversation:\n"
    for msg in history[-6:]:  # last 6 messages
        role = "User" if msg["type"] == "human" else "Assistant"
        context += f"{role}: {msg['content']}\n"

    return context

# 7. SAVE MEMORY
def save_session(session_id, user_input, response):
    if session_id not in store:
        store[session_id] = []

    store[session_id].append({"type": "human", "content": user_input})
    store[session_id].append({"type": "ai", "content": response})

    save_memory(store)

# 8. PROMPT (UPDATED)
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful assistant.

Use the conversation context to answer questions.
If the user asks about previous information (like name, preferences, etc),
look into the context.

{memory_context}

Rules:
- Be concise
- Do not hallucinate
- Say "I don't know" if not found
"""),
    ("human", "{input}")
])

# 9. CHAIN
llm_chain = prompt | llm | parser

# 10. ROUTER
def router(x):
    query = x["input"]
    session_id = x["session_id"]

    # TOOL ROUTING
    if "weather" in query.lower():
        loc = get_current_location.invoke({})
        weather = get_weather.invoke({"location": loc})
        return {"output": weather, "source": "weather"}

    elif "location" in query.lower():
        loc = get_current_location.invoke({})
        return {"output": loc, "source": "location"}

    # MEMORY CONTEXT INJECTION (MAIN FIX)
    memory_context = build_memory_context(session_id)

    answer = llm_chain.invoke({
        "input": query,
        "memory_context": memory_context
    })

    return {"output": answer, "source": "llm"}

# 11. CHAIN PIPELINE
chain = RunnableLambda(router)

# 12. RUN
if __name__ == "__main__":
    print("🚀 Memory-Enabled Agent Started")

    session_id = "user1"

    while True:
        query = input("\nEnter query: ")

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        try:
            response = chain.invoke({
                "input": query,
                "session_id": session_id
            })

            # SAVE MEMORY
            save_session(session_id, query, response["output"])

            print("\n===== RESPONSE =====")
            print("Answer :", response["output"])
            print("Source :", response["source"])

        except Exception as e:
            print("\n❌ Error:", str(e))
            
# Why this file is different from Agent_Pipeline_with_Memory.py:
# 1. Memory Storage: This version uses a JSON file to persist memory across sessions,
    #while the previous version only kept memory in-memory (lost on restart)
# 2. Memory Context Injection: The router now builds a memory context string and injects it into the LLM prompt, allowing the model to reference past interactions.
# 3. Response Key Fix: The router returns the LLM response under the key "output" instead of "answer" to be compatible with `RunnableWithMessageHistory`.
# 4. Simplified Router: The routing logic is streamlined to focus on tool calls and LLM invocation with memory context, removing the separate normalization step.

# Important Note: 
# The main fix is the injection of memory context into the LLM prompt, which allows the agent to utilize past conversation history when generating responses.
# Additionally, ensuring the response key is "output" is crucial for the memory wrapper to function correctly.
# * Earlier, memory was only saved in JSON but never properly used by the LLM during response generation.
# * The `{history}` placeholder passed raw chat messages, which the LLM treated as conversation noise instead of usable knowledge.
# * Because memory was not explicitly retrieved and structured, the model often failed to answer questions like “What is my name?”.
# * The fix was to manually build a `memory_context` from JSON and inject it directly into the prompt.
# * Now the system works because it follows **store → retrieve → inject → generate**, making memory explicitly available to the LLM.


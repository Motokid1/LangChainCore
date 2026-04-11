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
import os
import json
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

# 1. ENV
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


# 4. JSON MEMORY
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


def build_memory_context(session_id):
    if session_id not in store:
        return ""

    history = store[session_id]

    context = "Previous conversation:\n"
    for msg in history[-6:]:
        role = "User" if msg["type"] == "human" else "Assistant"
        context += f"{role}: {msg['content']}\n"

    return context


def save_session(session_id, user_input, response):
    if session_id not in store:
        store[session_id] = []

    store[session_id].append({"type": "human", "content": user_input})
    store[session_id].append({"type": "ai", "content": response})

    save_memory(store)


# 5. SYSTEM PROMPT (IMPORTANT CHANGE)
system_prompt = """
You are a helpful assistant.

You have access to tools:
- get_current_location
- get_weather

Use tools when required.

Use conversation context to answer memory-based questions.

If you don't know something, say "I don't know".

Be concise.
"""


# 6. CREATE AGENT (NO ROUTER, NO LCEL)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)


# 7. RUN LOOP
if __name__ == "__main__":
    print("🚀 Agent with Memory Started (create_agent)")

    session_id = "user2"

    while True:
        query = input("\nEnter query: ")

        if query.lower() == "exit":
            break

        # inject memory context into input
        memory_context = build_memory_context(session_id)

        full_input = f"""
Memory Context:
{memory_context}

User Query:
{query}
"""

        response = agent.invoke({
            "messages": [
                ("user", full_input)
            ]
        })

        # extract final answer
        final_answer = response["messages"][-1].content

        # save memory
        save_session(session_id, query, final_answer)

        print("\n===== RESPONSE =====")
        print("Answer:", final_answer)
        for msg in response["messages"]:
            if getattr(msg, "tool_calls", None):
                print(msg.tool_calls)
            
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


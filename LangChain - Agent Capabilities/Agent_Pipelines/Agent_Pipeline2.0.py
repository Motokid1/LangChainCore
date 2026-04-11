# Implemeneted using Router Architecture and LCEL (LangChain Execution Language)

# User Input
#    ↓
# Router
#    ↓
#  ┌───────────────┬───────────────┬───────────────┐
#  ↓               ↓               ↓
# Weather Tool   Location Tool   Guarded LLM

# 1. Imports
import os
import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.runnables import RunnableLambda

# 2. ENV
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# 3. LLM
llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0
)

# 4. TOOLS
@tool
def get_current_location() -> str:
    """Get current location"""
    res = requests.get("http://ip-api.com/json/").json()
    return f"{res.get('city')}, {res.get('country')}"


@tool
def get_weather(location: str) -> str:
    """Get weather"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url).json()

    if res.get("cod") != 200:
        return f"Weather error: {res.get('message')}"

    return f"{location}: {res['main']['temp']}°C, {res['weather'][0]['description']}"

# 5. GUARDED LLM
def guarded_llm(user_input: str):
    prompt = f"""
You are a controlled assistant.

Rules:
- Answer only general questions
- If unsure say "I don't know"
- Never attempt to answer weather or location queries
- You can use the following tools:
1. get_current_location() - returns current location as "City, Country" string
2. get_weather(location) - returns weather for a given location


Question: {user_input}
"""
    return llm.invoke(prompt).content

# 6. ROUTER (CORE LOGIC)
def router(user_input: str):
    q = user_input.lower()

    if "weather" in q:
        loc = get_current_location.invoke({})
        weather = get_weather.invoke({"location": loc})
        return {"answer": weather, "source": "weather"}

    elif "location" in q:
        loc = get_current_location.invoke({})
        return {"answer": loc, "source": "location"}

    else:
        return {"answer": guarded_llm(user_input), "source": "llm"}

# 7. LCEL
chain = RunnableLambda(router)

# 8. RUN
if __name__ == "__main__":
    while True:
        query = input("\nEnter query: ")
        if query.lower() == "exit":
            break

        res = chain.invoke(query)

        print("\n===== RESPONSE =====\n")
        print("Answer :", res["answer"])
        print("Source :", res["source"])
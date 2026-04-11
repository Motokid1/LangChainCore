# 1. Imports
import os
import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.tools import tool

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# 2. Load ENV
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    raise ValueError("❌ OPENWEATHER_API_KEY not found in .env")

# 3. Initialize LLM
llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
    temperature=0
)

parser = StrOutputParser()

# 4. TOOLS
@tool
def get_current_location() -> str:
    """Get current user location using IP"""
    try:
        res = requests.get("http://ip-api.com/json/", timeout=5).json()

        if res.get("status") != "success":
            return "Unable to fetch location"

        return f"{res.get('city')}, {res.get('country')}, {res.get('isp')}"

    except Exception as e:
        return f"Location error: {str(e)}"


@tool
def get_weather(location: str) -> str:
    """Get real-time weather"""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url, timeout=5).json()

        if res.get("cod") != 200:
            return f"Weather API Error: {res.get('message')}"

        temp = res["main"]["temp"]
        desc = res["weather"][0]["description"]

        return f"{location}: {temp}°C, {desc}"

    except Exception as e:
        return f"Weather error: {str(e)}"

# 5. Prompt (Guardrails)
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a controlled assistant.

Rules:
- Be concise (max 3 lines)
- Do not hallucinate
- Say "I don't know" if unsure
"""),
    ("human", "{input}")
])

# 6. LLM Chain (LCEL)
llm_chain = prompt | llm | parser

# 7. Router Logic
def router_lcel(x):
    query = x["input"].lower()

    # Weather flow
    if "weather" in query:
        location = get_current_location.invoke({})
        weather = get_weather.invoke({"location": location})

        return {
            "answer": weather,
            "source": "weather"
        }

    # Location flow
    elif "location" in query:
        location = get_current_location.invoke({})
        return {
            "answer": location,
            "source": "location"
        }

    # LLM flow
    else:
        answer = llm_chain.invoke({"input": x["input"]})

        return {
            "answer": answer,
            "source": "llm"
        }

# 8. FINAL LCEL CHAIN
chain = (
    RunnableLambda(lambda x: {"input": x})
    | RunnableLambda(router_lcel)
)

# 9. RUN
if __name__ == "__main__":
    print("🚀 LCEL Agent Started (type 'exit' to quit)")

    while True:
        query = input("\nEnter query: ")

        if query.lower() == "exit":
            print("👋 Exiting...")
            break

        try:
            response = chain.invoke(query)

            print("\n===== FINAL RESPONSE =====\n")
            print("Answer :", response["answer"])
            print("Source :", response["source"])

        except Exception as e:
            print("\n❌ Error:", str(e))
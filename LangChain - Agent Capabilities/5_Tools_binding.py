import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

# Define tool
@tool
def about_me() -> str:
    """Tells about Rohith"""
    return "Rohith is a good developer"

# Bind tool
llm_with_tools = llm.bind_tools([about_me])

from langchain_core.messages import HumanMessage, ToolMessage

# 1. First invocation (Ask the question)
messages = [HumanMessage(content="Who is Rohith?")]
ai_msg = llm_with_tools.invoke(messages)

print("AI's Initial Response:\n", ai_msg.content)
messages.append(ai_msg)

print("\nAI's Tool Calls:\n", ai_msg.tool_calls)

# 2. Check if the AI wants to use a tool
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        # Run the actual function
        result = about_me.invoke(tool_call["args"])
        
        # Add the tool's answer to the conversation history
        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

# 3. Second invocation (Give the tool result back to the LLM)
final_response = llm_with_tools.invoke(messages)

print("\nFinal Answer:\n", final_response.content)
# print("\nConversation History:")
# for msg in messages:
#     print(f"{msg.__class__.__name__}: {msg.content}")   

# When you bind a tool to an LLM, the model enters a specific "mode." 
# If it decides it needs a tool to answer your question, it stops generating 
# text and instead generates a structured request to use that tool.

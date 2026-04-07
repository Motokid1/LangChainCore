import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()

# 1. Setup Model and Tool
llm = ChatGroq(model="openai/gpt-oss-120b") # Using a standard Groq model name

@tool
def multiply(a: int, b: int) -> int:
    """Multiply numbers quickly"""
    return a * b

llm_with_tool = llm.bind_tools([multiply])

# 2. First Pass: The LLM decides to use the tool
messages = [HumanMessage(content="Multiply 4 and 6")]
ai_msg = llm_with_tool.invoke(messages)
messages.append(ai_msg)

print("AI Message Tool Calls:", ai_msg.tool_calls)

# 3. Execution: You manually run the function based on the LLM's request
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        # Execute the actual function
        result = multiply.invoke(tool_call["args"])
        
        # 4. Feed the result back to the LLM via ToolMessage
        messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

# 5. Final Pass: The LLM sees the tool result and generates text
final_response = llm_with_tool.invoke(messages)
print("\nFinal AI Response:", final_response.content)
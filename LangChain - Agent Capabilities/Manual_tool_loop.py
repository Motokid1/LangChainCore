import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-120b")

@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def tool_user() -> str: # Removed a, b because they aren't needed for this
    """Check who is currently using the tool"""
    return "Rohith is using the tool"

# Create a dictionary to map tool names to functions
tools_map = {"add": add, "tool_user": tool_user}

llm_with_tools = llm.bind_tools(list(tools_map.values()))

messages = [HumanMessage(content="What is 10 + 20? And who is using the tool?")]

# Step 1: Model response
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

# Step 2: Loop through ALL tool calls
if ai_msg.tool_calls:
    for tool_call in ai_msg.tool_calls:
        # Get the function from our map based on the name the LLM requested
        selected_tool = tools_map[tool_call["name"]]
        
        # Run the tool
        tool_output = selected_tool.invoke(tool_call["args"])
        
        # Create the ToolMessage
        messages.append(ToolMessage(
            content=str(tool_output),
            tool_call_id=tool_call["id"]
        ))

    # Step 3: Final response (Now the LLM has all the facts)
    final_response = llm_with_tools.invoke(messages)
    print("\nFinal Answer:\n", final_response.content)
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

llm = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq"
)

@tool(response_format="content_and_artifact")
def calculate(a: int, b: int):
    """Add numbers"""

    result = a + b

    return (
        f"The sum is {result}",     # LLM sees
        {"result": result}          # structured data
    )

llm_with_tools = llm.bind_tools([calculate])

response = llm_with_tools.invoke("Add 10 and 20")

print("\nTool Calls:", response.tool_calls)
# print("\nArtifact for Application:", response.artifact)

#What is the content and artifact?
#response_format="content_and_artifact" lets a tool return TWO outputs, 
#instead of one — one for the LLM to read, and one for your application to use programmatically.
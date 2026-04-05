from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_groq import ChatGroq

# 1️⃣ Schema
class Person(BaseModel):
    name: str
    age: int

# 2️⃣ Parser
parser = PydanticOutputParser(pydantic_object=Person)

# 3️⃣ LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b", 
    temperature=0
    )

# 4️⃣ Prompt with placeholder
prompt = ChatPromptTemplate.from_messages([
    ("system", "You MUST ONLY return valid JSON in this format:\n{format_instructions}"),
    ("human", "Generate a random person")
])

# 5️⃣ Inject instructions safely using partial
partial_prompt = prompt.partial(format_instructions=parser.get_format_instructions())

# 6️⃣ Chain
chain = partial_prompt | llm | parser

# 7️⃣ Invoke
result = chain.invoke({})

print(result)
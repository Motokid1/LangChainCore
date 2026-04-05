from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Initialize model
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

# Create prompt with 2 variables
prompt = PromptTemplate.from_template(
    "Explain {topic} in {level} level"
)

# Use partial → fix 'level'
partial_prompt = prompt.partial(level = "simple")

# Chain
chain = partial_prompt | llm | StrOutputParser()

# Now only pass 'topic'
result = chain.invoke({"topic": "LangChain"})

print(result)
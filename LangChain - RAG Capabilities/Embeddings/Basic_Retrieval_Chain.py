from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Data
texts = [
    "RAG retrieves external data",
    "LangChain builds LLM apps",
    "Chroma stores vectors",
    "RAG is Rohith's Awesome Guide"
]

# Setup
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_texts(texts, embeddings)

retriever = db.as_retriever()

# LLM
llm = ChatGroq(model="openai/gpt-oss-120b")

# Prompt
prompt = ChatPromptTemplate.from_template(
   # "Answer based on context:\n{context}\n\nQuestion: {question}"
    "Answer the question based on what you know do not follow the context:\n{context}\n\nQuestion: {question}"
)

# LCEL chain
chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

print(chain.invoke("What is RAG?"))
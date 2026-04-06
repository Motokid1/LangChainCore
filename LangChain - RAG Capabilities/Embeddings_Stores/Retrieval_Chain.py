from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

texts = [
    "RAG retrieves external knowledge",
    "LangChain helps build AI apps"
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_texts(texts, embeddings)
retriever = db.as_retriever()

llm = ChatGroq(model="openai/gpt-oss-120b")

prompt = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}"
)

chain = RunnableParallel(
    context=retriever,
    question=lambda x: x
) | prompt | llm | StrOutputParser()

print(chain.invoke("Explain RAG"))
# 1. SETUP
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-120b")

# 2. VECTOR DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

docs = [
    Document(page_content="RAG retrieves documents before answering."),
    Document(page_content="Embeddings convert text into vectors.")
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever(search_kwargs={"k": 2})
print("Retriever ready.")

# 3. FORMAT FUNCTION
def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)

# 4. RAG CHAIN (LCEL)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
    "Answer using context:\n{context}\n\nQuestion: {question}"
)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": lambda x: x
    }
    | prompt
    | llm
    | StrOutputParser()
)

# 5. RUN
print(rag_chain.invoke("What is RAG?"))
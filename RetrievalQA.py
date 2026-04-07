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
    Document(page_content="RAG combines retrieval and generation."),
    Document(page_content="Vector databases store embeddings.")
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever()

# 3. RETRIEVAL QA
from langchain_classic.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# 4. RUN
response = qa_chain.invoke("What is RAG?")
print(response)
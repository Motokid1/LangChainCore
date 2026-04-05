from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

texts = [
    "RAG retrieves data before answering",
    "Chroma is used for storing embeddings"
]

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_texts(texts, embeddings)
retriever = db.as_retriever()

llm = ChatGroq(model="openai/gpt-oss-120b")

prompt = ChatPromptTemplate.from_template(
    "Answer using context:\n{context}\n\nQuestion: {input}"
)

doc_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, doc_chain)

print(rag_chain.invoke({"input": "What is RAG?"}))
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 1. Sample docs
docs = [
    Document(page_content="RAG improves LLM using external knowledge."),
    Document(page_content="LangChain helps build LLM applications.")
]

# 2. Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Store in Chroma (persistent)
db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="./chroma_db"
)

# 4. Load again (simulate restart)
db2 = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

retriever = db2.as_retriever()

results = retriever.invoke("What is LangChain?")
print(results[0].page_content)

#Why are the documents stored in the chroma database?
# The documents are stored in the chroma database to enable efficient retrieval based on their vector representations
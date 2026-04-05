from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Data
texts = [
    "LangChain is used for LLM apps",
    "RAG improves answers using retrieval",
    "Chroma is a vector database"
]

# Embeddings 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create DB
db = Chroma.from_texts(texts, embeddings, persist_directory="./chroma_db")

# Query
results = db.similarity_search("What is RAG?")

for r in results:
    print(r.page_content)
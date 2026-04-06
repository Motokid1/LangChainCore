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
results = db.similarity_search("What is Chroma?")

for r in results:
    print(r.page_content)

#what is similarity search?
# Similarity search is a method used to find documents that are semantically similar to a given query.
# It works by comparing the vector representations of the query and the documents in the database, and

#What is from_texts?
# The `from_texts` method is used to create a Chroma database from a list
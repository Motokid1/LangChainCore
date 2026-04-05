from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Sample data
texts = [
    "LangChain is a framework for LLM applications",
    "FAISS enables fast similarity search",
    "RAG combines retrieval and generation",
    "Rohith is a software engineer",
    "Rohith loves working with LLMs", 
    "Rohith is based in India"
]

# Step 2: Load  embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Create FAISS index
db = FAISS.from_texts(texts, embeddings)

# Step 4: Query
query = "What is Rohith?"

results = db.similarity_search(query)

# Step 5: Display results
for i, res in enumerate(results):
    print(f"\nResult {i+1}:")
    print(res.page_content)
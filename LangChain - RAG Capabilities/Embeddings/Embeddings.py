from langchain_community.embeddings import HuggingFaceEmbeddings

# Load model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Query embedding
vector = embeddings.embed_query("What is LangChain?")

print("Vector length:", len(vector))
print("First 10 values:", vector[:10])
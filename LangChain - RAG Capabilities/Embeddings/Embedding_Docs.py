from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "LangChain is used for LLM apps",
    "FAISS is a vector database",
    "RAG combines retrieval and generation"
]

vectors = embeddings.embed_documents(documents)

# Print the number of vectors generated
print("Number of vectors:", len(vectors))

# Each vector should have the same dimension as the embedding model's output
print("Dimension of each vector:", len(vectors[0]))

# Print first 5 dimensions of the first vector
print("First vector:", vectors[0][:5], "...") 
 
print("All vectors:", vectors)

print("Embedding process completed successfully.")


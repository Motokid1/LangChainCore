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

print("Number of vectors:", len(vectors))
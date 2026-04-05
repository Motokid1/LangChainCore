from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 2: Sample text
text = """
LangChain is used for building LLM applications.
RAG improves answers by retrieving external data.
FAISS helps in similarity search.
Transformers power modern NLP systems.
"""

# Step 3: Semantic Chunker
splitter = SemanticChunker(embeddings)

# Step 4: Create chunks
docs = splitter.create_documents([text])

# Step 5: Print chunks
for i, doc in enumerate(docs):
    print(f"\nChunk {i+1}:\n", doc.page_content)
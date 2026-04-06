# 1. ENV SETUP
from dotenv import load_dotenv
load_dotenv()

# 2. IMPORTS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.retrievers import BM25Retriever

#Import using LangChain Classic
from langchain_classic.retrievers import EnsembleRetriever

# 3. LOAD MULTIPLE DOCUMENTS
# Folder: data/ (put .txt files here)
loader = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
docs = loader.load()

print(f"Loaded {len(docs)} documents")

# 4. ADD METADATA
for i, doc in enumerate(docs):
    doc.metadata["source"] = f"file_{i}"
    doc.metadata["type"] = "text"

# 5. SPLIT DOCUMENTS
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

split_docs = splitter.split_documents(docs)

print(f"Total chunks: {len(split_docs)}")

# 6. EMBEDDINGS (FREE)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 7. CHROMA VECTOR STORE
db = Chroma.from_documents(
    split_docs,
    embeddings,
    persist_directory="./chroma_db"
)

# 8. RETRIEVERS

# 1. Similarity Retriever
similarity_retriever = db.as_retriever(search_kwargs={"k": 3})

# 2. MMR Retriever
mmr_retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3}
)

# 3. BM25 Retriever (keyword-based)
bm25_retriever = BM25Retriever.from_documents(split_docs)

# 4. Ensemble Retriever (combine all)
ensemble_retriever = EnsembleRetriever(
    retrievers=[similarity_retriever, mmr_retriever, bm25_retriever],
    weights=[0.4, 0.3, 0.3]
)

# 9. LLM (GROQ)
llm = ChatGroq(model="openai/gpt-oss-120b")

# 10. PROMPT TEMPLATE
prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant.
Also include source references in your answer.
Answer from what you know, at the end include the provided context.

Context:
{context}

Question:
{question}
"""
)

# 11. FORMAT DOCUMENTS FUNCTION
def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}"
        for doc in docs
    )

# 12. LCEL RAG PIPELINE
rag_chain = (
    RunnableParallel(
        context=similarity_retriever | format_docs,
        question=lambda x: x
    )
    | prompt
    | llm
    | StrOutputParser()
)

# 13. RUN QUERY
query = "What is RAG?"

response = rag_chain.invoke(query)

print("\n===== FINAL ANSWER =====\n")
print(response)
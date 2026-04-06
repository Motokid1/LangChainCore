from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

loader1 = TextLoader("file1.txt")
loader2 = TextLoader("file2.txt")

docs = loader1.load() + loader2.load()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(docs, embeddings)

retriever = db.as_retriever()

print(retriever.invoke("search something"))
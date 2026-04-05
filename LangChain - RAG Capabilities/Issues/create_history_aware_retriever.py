from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# LLM
llm = ChatGroq(model="openai/gpt-oss-120b")

# Docs
docs = [Document(page_content="RAG improves LLM responses.")]
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(docs, embeddings)
retriever = db.as_retriever()

# Prompt
prompt = ChatPromptTemplate.from_template(
    "Given chat history and question, make standalone question:\n{input}"
)

# Create history-aware retriever
history_retriever = create_history_aware_retriever(
    llm, retriever, prompt
)

result = history_retriever.invoke({
    "input": "How does it help?",
    "chat_history": []
})

print(result)
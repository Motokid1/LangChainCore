# 1. SETUP
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-120b")

# 2. VECTOR DB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

docs = [
    Document(page_content="RAG improves LLM responses using retrieval."),
    Document(page_content="LangChain helps build LLM applications."),
    Document(page_content="JD is Job Description.")
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings)

# 3. MANUAL SEARCH
# query = "What is RAG?"

results = db.similarity_search(query = "What is JD?", k=2)

print("Search Results:")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content}")

# context = "\n".join([doc.page_content for doc in results])

# # 4. PROMPT + LLM
# from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_template(
#     "Answer using context:\n{context}\n\nQuestion: {question}"
# )

# chain = prompt | llm

# response = chain.invoke({"context": context, "question": query})

# print(response.content)
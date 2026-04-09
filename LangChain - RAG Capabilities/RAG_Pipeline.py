# 1. LOAD ENV + LLM
from dotenv import load_dotenv
load_dotenv()


from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-120b")  


# 2. LOAD DOCUMENTS (TXT + PDF)
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader
)

docs = []

txt_loader = DirectoryLoader(
    "data/", glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs.extend(txt_loader.load())

pdf_loader = DirectoryLoader(
   "D:\Gen AI\LangChain - RAG Capabilities\data",
    glob="**/*.pdf", 
    loader_cls=PyPDFLoader
    )
docs.extend(pdf_loader.load())

print(f"Loaded documents: {len(docs)}")

if not docs:
    raise ValueError("No documents found in data/ folder!")


# 3. SPLIT DOCUMENTS
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)
split_docs = [doc for doc in split_docs if doc.page_content.strip()]

print(f"Total chunks: {len(split_docs)}")

if not split_docs:
    raise ValueError("No valid chunks!")


# 4. EMBEDDINGS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 5. CHROMA VECTOR STORE
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(split_docs, embeddings, persist_directory="./chroma_db")
retriever = db.as_retriever(search_kwargs={"k": 3})


# 6. QA PROMPT
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use the retrieved context to answer the question.
If answer is not found, say I don't know and Fetch from NASA database.
Include source references at the end.

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),  # ✅ Proper chat history placeholder
    ("human", "{input}")
])

from langsmith import traceable


# 7. FORMAT DOCUMENTS
@traceable  # ✅ This will now show up as a clear step in LangSmith
def format_docs(docs):
    return "\n\n".join(
        f"{doc.page_content}\nSOURCE: {doc.metadata.get('source', 'unknown')}"
        for doc in docs
    )


# 8. LCEL RAG CHAIN
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# ✅ Key fix: extract string from dict before retriever
@traceable
def get_input_str(x):
    return x["input"] if isinstance(x, dict) else x

#what does get_input_str do?
#It checks if the input is a dictionary and extracts the "input" key's value. 
#If it's not a dictionary, it returns the input as is. 
#This ensures that the retriever receives a clean string query, preventing errors when the input is passed through the chain.

#why is it necessary?
#In the RAG chain, the input is often passed as a dictionary containing various keys 
#(like "input" for the query and "chat_history" for previous interactions).
#If we pass the entire dictionary to the retriever, it will cause an error since the retriever expects a string query.
#By using get_input_str, we ensure that only the relevant query string is sent to the retriever, allowing the chain to function correctly without type errors.

rag_chain = (
    {
        "context": RunnableLambda(get_input_str) | retriever | format_docs,
        "input": lambda x: x["input"],
        "chat_history": lambda x: x.get("chat_history", [])
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)


# 9. MEMORY
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

rag_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


# 10. CHAT LOOP
print("\n✅ RAG System Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    response = rag_with_memory.invoke(
        {"input": query},
        config={"configurable": {"session_id": "user1"}}
    )

    print("\nAI:", response, "\n")
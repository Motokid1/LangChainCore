from dotenv import load_dotenv  
load_dotenv()  
from langchain_groq import ChatGroq  

llm = ChatGroq(
    model="openai/gpt-oss-120b",  
)

from langchain_community.document_loaders import (
    DirectoryLoader,   # Scans a folder and loads all matching files
    TextLoader,        # Loads plain .txt files
    PyPDFLoader,       # Loads PDF files page by page
    WebBaseLoader      # Fetches and loads content from URLs
)

docs = []  

# ── TXT Files ──
txt_loader = DirectoryLoader(
    path = "D:/Gen AI/LangChain - RAG Capabilities/data",                       
    glob="**/*.txt",                  
    loader_cls=TextLoader,                
    loader_kwargs={"encoding": "utf-8"}  
)
docs.extend(txt_loader.load())        
# .load() returns a list of Document objects
# .extend() adds them all into our docs list

# ── PDF Files ──
pdf_loader = DirectoryLoader(
    path = "D:/Gen AI/LangChain - RAG Capabilities/data",
    glob="**/*.pdf",                  
    loader_cls=PyPDFLoader            
)
docs.extend(pdf_loader.load())        
# Append PDF docs to the same list

# ── Web URLs ──
web_loader = WebBaseLoader([
    "https://google.com",            
])
docs.extend(web_loader.load())        
# Append web docs to the same list

print(f"Total documents loaded: {len(docs)}")  
# Quick sanity check — should be > 0

if not docs:
    raise ValueError("No documents found! Check your data/ folder and URLs.")
    # Stop execution early if nothing loaded — better than a confusing error later


from langchain_text_splitters import RecursiveCharacterTextSplitter  
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    
    chunk_overlap=50   
)

split_docs = splitter.split_documents(docs)  
split_docs = [doc for doc in split_docs if doc.page_content.strip()]  

print(f"Total chunks after splitting: {len(split_docs)}")

if not split_docs:
    raise ValueError("No valid chunks after splitting!")

# STEP 5: CREATE EMBEDDINGS
from langchain_huggingface import HuggingFaceEmbeddings  

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  
)


# STEP 6: STORE EMBEDDINGS IN CHROMA
from langchain_community.vectorstores import Chroma  
db = Chroma.from_documents(
    split_docs,                   
    embeddings,                   
    persist_directory="./chroma_db"  
)

retriever = db.as_retriever(
    search_kwargs={"k": 3}   
    # k=3 means fetch the top 3 most similar chunks for each query
    # Increase k for more context, decrease for less noise
)


# STEP 7: BUILD THE PROMPT TEMPLATE
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
# ChatPromptTemplate builds structured prompts with roles
# MessagesPlaceholder is a slot where chat history gets injected

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
Use the retrieved context below to answer the question.
If the answer is not in the context, say 'I don't know'.
Always include source references at the end of your answer.

Context:
{context}"""),
    # system message sets the AI's behavior
    # {context} will be replaced with retrieved chunks at runtime

    MessagesPlaceholder("chat_history"),
    # This slot gets filled with previous conversation messages
    # Enables multi-turn conversation memory

    ("human", "{input}")
    # {input} will be replaced with the user's current question
])


# STEP 8: HELPER — FORMAT RETRIEVED DOCS
def format_docs(docs):
    # Takes a list of Document objects returned by the retriever
    # Returns a single formatted string with content + source
    return "\n\n".join(
        f"{doc.page_content}\nSOURCE: {doc.metadata.get('source', 'unknown')}"
        # page_content = the actual text chunk
        # metadata['source'] = file path or URL it came from
        for doc in docs
    )


# STEP 9: BUILD THE RAG CHAIN
from langchain_core.output_parsers import StrOutputParser  
# Converts the LLM's AIMessage object into a plain Python string

from langchain_core.runnables import RunnableLambda  
# Wraps a plain Python function so it can be used inside LCEL chains

def get_input_str(x):
    # The retriever expects a plain string, but our chain passes a dict
    # This function extracts just the "input" string from the dict
    return x["input"] if isinstance(x, dict) else x

rag_chain = (
    {
        "context": RunnableLambda(get_input_str) | retriever | format_docs,
        # 1. Extract input string from dict
        # 2. Pass to retriever → returns top 3 relevant Document chunks
        # 3. format_docs() formats them into one readable string

        "input": lambda x: x["input"],
        # Pass the raw question through to fill {input} in prompt

        "chat_history": lambda x: x.get("chat_history", [])
        # Pass chat history through — empty list if first message
    }
    | qa_prompt      
    # Fills {context}, {input}, {chat_history} into the prompt template
    
    | llm            
    # Sends the filled prompt to Groq → returns AIMessage
    
    | StrOutputParser()  
    # Extracts .content from AIMessage → returns plain string
)


# STEP 10: ADD MEMORY / CHAT HISTORY

from langchain_core.runnables.history import RunnableWithMessageHistory  
# Wraps any chain to automatically inject + update chat history

from langchain_community.chat_message_histories import ChatMessageHistory  
# In-memory store for one conversation session
# Stores HumanMessage and AIMessage objects

store = {}  
# Dictionary to hold multiple sessions: { session_id: ChatMessageHistory }

def get_session_history(session_id: str):
    # Called automatically by RunnableWithMessageHistory before each invoke
    # Returns the history object for the given session
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
        # Create a new empty history if this session is new
    return store[session_id]

rag_with_memory = RunnableWithMessageHistory(
    rag_chain,                          
    # The chain to wrap with memory
    
    get_session_history,                
    # Function that returns history for a given session ID
    
    input_messages_key="input",         
    # Which key in the input dict holds the user's message
    
    history_messages_key="chat_history" 
    # Which key in the chain expects the history to be injected into
)


# STEP 11: CHAT LOOP

print("\n✅ RAG System Ready! Type 'exit' to quit.\n")

while True:
    query = input("You: ")  
    # Wait for user to type a question

    if query.lower() == "exit":
        # .lower() makes the check case-insensitive (Exit, EXIT all work)
        print("Goodbye!")
        break  
        # Exit the loop and end the program

    response = rag_with_memory.invoke(
        {"input": query},   
        # Pass user's question as a dict
        
        config={"configurable": {"session_id": "user1"}}
        # session_id identifies which conversation history to use
        # Change "user1" to support multiple users simultaneously
    )

    print(f"\nAI: {response}\n")  
    # Print the final string response from StrOutputParser
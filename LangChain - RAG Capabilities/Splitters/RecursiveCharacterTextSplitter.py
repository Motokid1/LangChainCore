from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "LangChain is a framework for building LLM applications." * 50

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for chunk in chunks:
    print(chunk)
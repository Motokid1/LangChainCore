from langchain_text_splitters import TokenTextSplitter

text = "LangChain enables building AI apps." * 50

splitter = TokenTextSplitter(
    chunk_size=50,
    chunk_overlap=10
)

chunks = splitter.split_text(text)

print(chunks)
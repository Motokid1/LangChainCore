from langchain_text_splitters import RecursiveCharacterTextSplitter

text = [
    "LangChain is a framework for building LLM applications.",
    "It provides tools for working with language models, including text splitting.",
    "The RecursiveCharacterTextSplitter is a powerful tool for splitting text into manageable chunks while preserving context.",
    "It allows you to specify the chunk size and overlap, making it ideal for processing large documents."
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text([t for t in text if t.strip()])

for chunk in chunks:
    print(chunk)
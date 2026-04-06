from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader("data/", glob="*.txt" )

docs = loader.load()

print(f"Loaded {len(docs)} docs")
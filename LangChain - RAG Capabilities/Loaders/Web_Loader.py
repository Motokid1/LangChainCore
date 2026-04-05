from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Computer")

docs = loader.load()

print(docs[0].page_content[:500])
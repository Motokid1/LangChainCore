from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("sample.pdf")

docs = loader.load()

for doc in docs:
    print(doc.page_content[:200])
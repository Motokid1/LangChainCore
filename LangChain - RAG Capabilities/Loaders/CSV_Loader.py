from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="data.csv")

docs = loader.load()

for doc in docs:
    print(doc.page_content)
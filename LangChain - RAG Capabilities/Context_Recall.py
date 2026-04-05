retrieved_docs = [
    "RAG improves LLM",
    "LangChain builds apps"
]

ground_truth = "RAG improves LLM performance"

# simple recall check
score = sum([1 for doc in retrieved_docs if "RAG" in doc]) / len(retrieved_docs)

print("Context Recall:", score)
# Ground truth is the ideal/expected answer
ground_truth = "RAG improves LLM performance"

# These are the chunks retrieved from your vector DB
retrieved_docs = [
    "RAG improves LLM",
    "LangChain builds apps",
    "LLM performance is enhanced by retrieval"
]

# ---------- Step 1: Break ground truth into key facts ----------
# We split the ground truth into individual words as "atomic facts"
# In real RAG eval tools like RAGAS, an LLM does this splitting smartly
ground_truth_facts = set(ground_truth.lower().split())
print("Facts to cover:", ground_truth_facts)
# {'rag', 'improves', 'llm', 'performance'}

# ---------- Step 2: Collect all words from retrieved docs ----------
# Merge all retrieved chunks into one big pool of words
retrieved_text = " ".join(retrieved_docs).lower()
retrieved_words = set(retrieved_text.split())
print("Retrieved words:", retrieved_words)

# ---------- Step 3: Find which facts were covered ----------
# Check which ground truth facts appear in the retrieved docs
covered_facts = ground_truth_facts.intersection(retrieved_words)
print("Covered facts:", covered_facts)
# {'rag', 'improves', 'llm', 'performance'} → all 4 covered

# ---------- Step 4: Calculate Recall ----------
# Recall = covered facts / total facts in ground truth
recall_score = len(covered_facts) / len(ground_truth_facts)
print("Context Recall:", round(recall_score, 2))
# Context Recall: 1.0 → perfect recall


#Context Recall measures what percentage of the information present in the 
# ground truth was successfully retrieved by the retriever.

# Context Recall = Relevant chunks retrieved
#                  ─────────────────────────
#                  Total chunks in ground truth
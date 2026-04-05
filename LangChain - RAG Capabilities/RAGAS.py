from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from dotenv import load_dotenv
load_dotenv()

# ================================
# 1. DATA
# ================================
data = {
    "question": ["What is RAG?"],
    "answer": ["RAG improves LLM using external knowledge"],
    "contexts": [["RAG improves LLM using external knowledge"]],
    "ground_truth": ["RAG improves LLM performance"]
}

dataset = Dataset.from_dict(data)

# ================================
# 2. USE GROQ AS LLM (Free)
# ================================
groq_llm = LangchainLLMWrapper(ChatGroq(model="llama3-8b-8192"))

# ================================
# 3. USE HUGGINGFACE EMBEDDINGS (Free)
# ================================
hf_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)

# ================================
# 4. DEFINE METRICS
# ================================
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
]

# ================================
# 5. EVALUATE
# ================================
result = evaluate(
    dataset,
    metrics=metrics,
    llm=groq_llm,
    embeddings=hf_embeddings
)

print("\n===== RAGAS EVALUATION RESULT =====\n")
print(result)
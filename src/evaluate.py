import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/deepresearch/.env"))
INDEX_DIR = os.path.expanduser("~/deepresearch/index")

TEST_QUERIES = [
    {"query": "what is retrieval augmented generation", "relevant_keywords": ["retrieval", "augmented", "generation", "rag"]},
    {"query": "how does LoRA fine tuning work", "relevant_keywords": ["lora", "fine-tuning", "low-rank", "adaptation"]},
    {"query": "what causes hallucination in language models", "relevant_keywords": ["hallucination", "factual", "generation", "error"]},
    {"query": "how does FAISS vector search work", "relevant_keywords": ["faiss", "vector", "search", "index", "embedding"]},
    {"query": "what is parameter efficient fine tuning", "relevant_keywords": ["parameter", "efficient", "fine-tuning", "peft"]},
    {"query": "how do transformer attention mechanisms work", "relevant_keywords": ["attention", "transformer", "self-attention", "head"]},
    {"query": "what are the limitations of large language models", "relevant_keywords": ["limitation", "llm", "language", "model"]},
    {"query": "how to evaluate RAG system performance", "relevant_keywords": ["evaluation", "metric", "rouge", "precision"]},
    {"query": "what is dense passage retrieval", "relevant_keywords": ["dense", "passage", "retrieval", "embedding"]},
    {"query": "how does instruction tuning improve language models", "relevant_keywords": ["instruction", "tuning", "fine-tuning", "prompt"]},
]

def precision_at_5(results, relevant_keywords):
    relevant = 0
    for chunk in results[:5]:
        text_lower = chunk["text"].lower()
        if any(kw in text_lower for kw in relevant_keywords):
            relevant += 1
    return relevant / 5

def evaluate():
    print("Loading model and index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(os.path.join(INDEX_DIR, "papers.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    print(f"\nRunning evaluation on {len(TEST_QUERIES)} queries...\n")
    scores = []
    for i, test in enumerate(TEST_QUERIES):
        query_embedding = model.encode([test["query"]]).astype("float32")
        distances, indices = index.search(query_embedding, 5)
        results = [metadata[idx] for idx in indices[0] if idx < len(metadata)]
        score = precision_at_5(results, test["relevant_keywords"])
        scores.append(score)
        print(f"[{i+1:02d}] P@5={score:.2f} | {test['query']}")

    avg = sum(scores) / len(scores)
    print(f"\n{'='*50}")
    print(f"Average Precision@5: {avg:.2f}")
    print(f"{'='*50}")
    if avg >= 0.70:
        print("✅ PASS — meets the 0.70 target from your project objectives!")
    else:
        print(f"⚠️  Below 0.70 target — need to improve retrieval")

if __name__ == "__main__":
    evaluate()

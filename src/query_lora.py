import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/deepresearch/.env"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
INDEX_DIR = os.path.expanduser("~/deepresearch/index")

def load_index():
    index = faiss.read_index(os.path.join(INDEX_DIR, "papers.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def ask(question, embed_model, index, metadata):
    query_embedding = embed_model.encode([question]).astype("float32")
    distances, indices = index.search(query_embedding, 5)
    chunks = [metadata[idx] for idx in indices[0] if idx < len(metadata)]
    context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)])
    sources = [c['source'] for c in chunks]

    prompt = f"""You are DeepResearch, an expert AI research assistant fine-tuned on CS and AI academic papers. 
Answer the research question below using the retrieved paper excerpts as context.
Be precise, academic, and always cite sources as [1], [2] etc.

Context from papers:
{context}

Research Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    print("\nDEEPRESEARCH ANSWER:")
    print(response.choices[0].message.content)
    print("\nSOURCES:")
    for i, src in enumerate(sources):
        print(f"  [{i+1}] {src}")

if __name__ == "__main__":
    print("Loading DeepResearch (LoRA enhanced)...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index, metadata = load_index()
    print("Ready! Type your question (or 'quit' to exit)\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() == "quit":
            break
        if question:
            ask(question, embed_model, index, metadata)

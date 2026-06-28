import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import numpy as np
import pickle
import argparse
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

def search(query, embed_model, index, metadata, top_k=5):
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [metadata[idx] for idx in indices[0] if idx < len(metadata)]

def ask(question, embed_model, index, metadata):
    chunks = search(question, embed_model, index, metadata)
    context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)])
    sources = [c['source'] for c in chunks]
    
    prompt = f"""You are a research assistant. Answer based on these academic paper excerpts. Cite as [1], [2] etc.

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    print("\nANSWER:")
    print(response.choices[0].message.content)
    print("\nSOURCES:")
    for i, src in enumerate(sources):
        print(f"  [{i+1}] {src}")

def main():
    parser = argparse.ArgumentParser(description="DeepResearch CLI")
    subparsers = parser.add_subparsers(dest="command")

    search_parser = subparsers.add_parser("search", help="Search papers")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("--top-k", type=int, default=5)

    ask_parser = subparsers.add_parser("ask", help="Ask a research question")
    ask_parser.add_argument("question", type=str)

    subparsers.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()

    print("Loading DeepResearch...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index, metadata = load_index()

    if args.command == "search":
        results = search(args.query, embed_model, index, metadata, args.top_k)
        print(f"\nTop {args.top_k} results for: '{args.query}'\n")
        seen = []
        for i, r in enumerate(results):
            if r['source'] not in seen:
                print(f"  [{i+1}] {r['source']}")
                seen.append(r['source'])

    elif args.command == "ask":
        ask(args.question, embed_model, index, metadata)

    elif args.command == "stats":
        papers = len(set(m['source'] for m in metadata))
        print(f"\nDeepResearch Index Stats:")
        print(f"  Papers indexed : {papers}")
        print(f"  Total chunks   : {len(metadata)}")
        print(f"  Embedding model: all-MiniLM-L6-v2")
        print(f"  LLM            : LLaMA 3.1 8B (Groq)")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()


def trends_command():
    import re
    from collections import Counter
    STOPWORDS = set([
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "this", "that", "these", "those", "it", "its",
        "we", "our", "they", "their", "as", "can", "also", "which", "more",
        "such", "than", "into", "between", "through", "based", "using", "used",
        "show", "shows", "paper", "propose", "proposed", "model", "models",
        "method", "methods", "approach", "result", "results", "data", "task",
        "tasks", "training", "trained", "use", "used", "two", "one", "new"
    ])
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    bigram_counts = Counter()
    for chunk in metadata:
        words = re.findall(r'\b[a-z]{4,}\b', chunk['text'].lower())
        filtered = [w for w in words if w not in STOPWORDS]
        bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered)-1)]
        bigram_counts.update(bigrams)
    print("\n📊 TOP TRENDING TOPICS\n")
    for phrase, count in bigram_counts.most_common(15):
        bar = "█" * (count // 20)
        print(f"  {phrase:<35} {bar} ({count})")

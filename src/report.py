import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import numpy as np
import pickle
from datetime import datetime
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

def retrieve(query, embed_model, index, metadata, top_k=8):
    query_embedding = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [metadata[idx] for idx in indices[0] if idx < len(metadata)]

def generate_report(question, embed_model, index, metadata):
    print(f"\nGenerating research report for: '{question}'")
    print("Retrieving relevant papers...")
    chunks = retrieve(question, embed_model, index, metadata)
    context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)])
    sources = list(dict.fromkeys([c['source'] for c in chunks]))

    prompt = f"""You are an expert research assistant. Generate a structured research report answering the question below, based on the academic paper excerpts provided.

Structure your report as follows:
1. OVERVIEW — brief answer to the question (2-3 sentences)
2. KEY FINDINGS — 3-5 bullet points of the most important findings from the papers
3. TECHNICAL DETAILS — deeper explanation of the technical concepts involved
4. LIMITATIONS & OPEN PROBLEMS — what the papers say is still unsolved
5. CONCLUSION — 2-3 sentence summary

Cite sources as [1], [2] etc throughout.

Context from papers:
{context}

Question: {question}

Report:"""

    print("Generating report with LLaMA 3.1...\n")
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )

    report_text = response.choices[0].message.content

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"
    output_path = os.path.join(os.path.expanduser("~/deepresearch"), filename)
    
    with open(output_path, "w") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"**Question:** {question}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Papers searched:** 226\n\n")
        f.write("---\n\n")
        f.write(report_text)
        f.write("\n\n---\n\n## Sources\n\n")
        for i, src in enumerate(sources):
            f.write(f"- [{i+1}] {src}\n")

    print(report_text)
    print(f"\n✅ Report saved to: {filename}")

if __name__ == "__main__":
    import sys
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "what is retrieval augmented generation?"
    print("Loading DeepResearch...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index, metadata = load_index()
    generate_report(question, embed_model, index, metadata)

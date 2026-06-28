import arxiv
import os
import requests
import time

SAVE_DIR = os.path.expanduser("~/deepresearch/papers")

def fetch_papers():
    client = arxiv.Client()
    queries = [
        "retrieval augmented generation",
        "large language model hallucination",
        "LoRA fine tuning language models",
        "FAISS vector search embeddings",
        "local language model deployment",
        "transformer architecture attention",
        "parameter efficient fine tuning",
        "natural language processing NLP"
    ]
    
    total = 0
    for query in queries:
        print(f"\nFetching: {query}")
        search = arxiv.Search(query=query, max_results=25, sort_by=arxiv.SortCriterion.Relevance)
        for result in client.results(search):
            paper_id = result.entry_id.split("/")[-1]
            pdf_path = os.path.join(SAVE_DIR, f"{paper_id}.pdf")
            if os.path.exists(pdf_path):
                continue
            try:
                response = requests.get(result.pdf_url, timeout=30)
                with open(pdf_path, "wb") as f:
                    f.write(response.content)
                total += 1
                print(f"  [{total}] {result.title[:60]}...")
                time.sleep(0.3)
            except Exception as e:
                print(f"  Failed: {e}")
    
    all_pdfs = [f for f in os.listdir(SAVE_DIR) if f.endswith(".pdf")]
    print(f"\nDone! Total papers: {len(all_pdfs)}")

if __name__ == "__main__":
    fetch_papers()

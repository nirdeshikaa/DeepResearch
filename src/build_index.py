import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import gc
import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

PAPERS_DIR = os.path.expanduser("~/deepresearch/papers")
INDEX_DIR = os.path.expanduser("~/deepresearch/index")

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        fitz.TOOLS.store_shrink(100)
        return " ".join(pages)
    except:
        return ""

def chunk_text(text, chunk_size=256, overlap=32):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks[:15]

def build_index():
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded!")

    all_embeddings = []
    all_metadata = []

    pdfs = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
    print(f"Processing {len(pdfs)} papers...")

    for i, pdf_file in enumerate(pdfs):
        pdf_path = os.path.join(PAPERS_DIR, pdf_file)
        text = extract_text(pdf_path)
        if not text:
            print(f"[{i+1}/{len(pdfs)}] SKIPPED {pdf_file}")
            continue
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = model.encode(chunks, batch_size=1, show_progress_bar=False)
        all_embeddings.append(np.array(embeddings, dtype="float32"))
        for chunk in chunks:
            all_metadata.append({"source": pdf_file, "text": chunk})
        print(f"[{i+1}/{len(pdfs)}] OK — {pdf_file[:40]} ({len(chunks)} chunks)")
        del text, chunks, embeddings
        gc.collect()

    print("Building FAISS index...")
    all_embeddings = np.vstack(all_embeddings).astype("float32")
    dimension = all_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(all_embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, "papers.index"))
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"Done! Indexed {len(all_metadata)} chunks from {len(pdfs)} papers.")

if __name__ == "__main__":
    build_index()

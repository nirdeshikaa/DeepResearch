# DeepResearch 🔬

> AI-powered research paper discovery, synthesis, and question answering — running locally.

DeepResearch fetches CS/AI papers from arXiv, builds a semantic search index, and lets you ask research questions to receive LLM-synthesised, citation-backed answers.

## Features
| Capability | Description |
|---|---|
| **Ingest** | Fetch arXiv papers, extract text, create embeddings |
| **Semantic Search** | Find relevant chunks with FAISS cosine similarity |
| **Ask** | RAG-based question answering with citations |
| **Report** | Full structured research report generator |
| **Trends** | Surface trending topics across indexed papers |
| **Evaluate** | Precision@5 evaluation — achieved 0.96 on 10 queries |

## Tech Stack
- **FAISS** — semantic vector search
- **sentence-transformers** — all-MiniLM-L6-v2 embeddings
- **LLaMA 3.1 8B** via Groq — answer generation
- **PyMuPDF** — PDF text extraction
- **arXiv API** — paper ingestion

## Usage
```bash
python src/cli.py stats
python src/cli.py search "LoRA fine tuning"
python src/cli.py ask "what is retrieval augmented generation?"
python src/report.py "how does LoRA reduce memory requirements?"
python src/evaluate.py
python src/trends.py
```

## Results
- 226 CS/AI papers indexed
- 3,337 searchable chunks
- Precision@5: **0.96** (target: 0.70) ✅

## Student
- **Name:** Nirdeshika Pandey
- **ID:** 23189630
- **Course:** BSc Hons Computer and Data Science
- **University:** Birmingham City University

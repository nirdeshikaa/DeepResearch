import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import fitz
import json
import time
import random
from groq import Groq
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/deepresearch/.env"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PAPERS_DIR = os.path.expanduser("~/deepresearch/papers")
OUTPUT_FILE = os.path.expanduser("~/deepresearch/training_data.json")

def extract_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        # Get first 400 words
        words = text.split()
        return " ".join(words[:400]) if len(words) > 50 else ""
    except:
        return ""

def generate_qa(text, source):
    prompt = f"""You are an expert in CS and AI research. Given this paper excerpt, generate 3 question-answer pairs about the technical content.

Paper excerpt:
{text}

Generate exactly 3 question-answer pairs in this JSON format:
[
  {{"instruction": "specific technical question", "response": "detailed technical answer"}},
  {{"instruction": "specific technical question", "response": "detailed technical answer"}},
  {{"instruction": "specific technical question", "response": "detailed technical answer"}}
]

Return ONLY the JSON array, nothing else."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        text_response = response.choices[0].message.content.strip()
        start = text_response.find("[")
        end = text_response.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        pairs = json.loads(text_response[start:end])
        for p in pairs:
            p["source"] = source
        return pairs
    except Exception as e:
        print(f"  Parse error: {e}")
        return []

def generate_dataset():
    pdfs = [f for f in os.listdir(PAPERS_DIR) if f.endswith(".pdf")]
    random.shuffle(pdfs)

    # Test first PDF
    test_text = extract_text(os.path.join(PAPERS_DIR, pdfs[0]))
    print(f"Test extraction: {len(test_text.split())} words from {pdfs[0]}")
    print(f"Sample: {test_text[:200]}\n")

    dataset = []
    target = 500

    print(f"Generating dataset from {len(pdfs)} papers. Target: {target} examples\n")

    for i, pdf_file in enumerate(pdfs):
        if len(dataset) >= target:
            break

        pdf_path = os.path.join(PAPERS_DIR, pdf_file)
        text = extract_text(pdf_path)
        if not text:
            print(f"[{i+1}] SKIPPED {pdf_file} — no text")
            continue

        pairs = generate_qa(text, pdf_file)
        if pairs:
            dataset.extend(pairs)
            print(f"[{i+1}/{len(pdfs)}] {pdf_file[:40]} — +{len(pairs)} (total: {len(dataset)})")
        else:
            print(f"[{i+1}/{len(pdfs)}] {pdf_file[:40]} — failed to generate")

        time.sleep(1.5)

    dataset = dataset[:target]

    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"\n✅ Done! {len(dataset)} examples saved to training_data.json")

if __name__ == "__main__":
    generate_dataset()

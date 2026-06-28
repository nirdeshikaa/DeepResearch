import os
import pickle
from collections import Counter
import re

INDEX_DIR = os.path.expanduser("~/deepresearch/index")

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

def get_trends(top_n=20):
    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    word_counts = Counter()
    bigram_counts = Counter()

    for chunk in metadata:
        words = re.findall(r'\b[a-z]{4,}\b', chunk['text'].lower())
        filtered = [w for w in words if w not in STOPWORDS]
        word_counts.update(filtered)
        bigrams = [f"{filtered[i]} {filtered[i+1]}" for i in range(len(filtered)-1)]
        bigram_counts.update(bigrams)

    print("\n📊 TOP TRENDING TOPICS IN YOUR PAPER LIBRARY\n")
    print("Single Keywords:")
    for word, count in word_counts.most_common(top_n):
        bar = "█" * (count // 50)
        print(f"  {word:<25} {bar} ({count})")

    print("\nKey Phrases:")
    for phrase, count in bigram_counts.most_common(top_n):
        bar = "█" * (count // 20)
        print(f"  {phrase:<35} {bar} ({count})")

if __name__ == "__main__":
    get_trends()

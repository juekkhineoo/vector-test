"""
Demo: embed a small corpus and run similarity search.

Run:
    python main.py
"""

from embedder import TextEmbedder
from vector_store import VectorStore

# ---------------------------------------------------------------------------
# Sample corpus
# ---------------------------------------------------------------------------
CORPUS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "Neural networks are inspired by the human brain.",
    "Natural language processing deals with text and speech.",
    "Vector databases store high-dimensional embeddings efficiently.",
    "Transformers revolutionized the field of NLP.",
    "FAISS is a library for efficient similarity search.",
    "Sentence embeddings capture semantic meaning of text.",
    "HuggingFace provides pre-trained models for various NLP tasks.",
    "Chiang Mai is a city in northern Thailand known for its temples.",
    "The Great Wall of China is a historic fortification built to protect against invasions.",
]

QUERIES = [
    "How do AI models understand language?",
    "What tools help with fast vector search?",
    "Tell me about Python for data analysis.",
    "What is processing natural language?",
    "Tell me about Chiang Mai.",
    "What is the Great Wall of China?",
]


def main():
    # 1. Load model
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")

    # 2. Build vector store
    store = VectorStore(embedding_dim=embedder.embedding_dim, metric="cosine")

    print("\n--- Embedding corpus ---")
    corpus_vectors = embedder.embed(CORPUS, show_progress=True)
    store.add(
        vectors=corpus_vectors,
        texts=CORPUS,
        metadata=[{"source": "demo", "index": i} for i in range(len(CORPUS))],
    )
    print(store)

    # 3. Similarity search
    print("\n--- Similarity Search ---")
    for query in QUERIES:
        print(f"\nQuery : {query!r}")
        query_vec = embedder.embed(query)
        results = store.search(query_vec, top_k=3)
        for rank, result in enumerate(results, 1):
            print(f"  [{rank}] score={result.score:.4f}  \"{result.text}\"")

    # 4. Pairwise similarity example
    print("\n--- Pairwise Similarity ---")
    pairs = [
        ("I love machine learning.", "Deep learning is amazing."),
        ("I love machine learning.", "The weather is nice today."),
    ]
    for a, b in pairs:
        va, vb = embedder.embed(a), embedder.embed(b)
        sim = embedder.similarity(va, vb)
        print(f"  {sim:.4f}  |  {a!r}  vs  {b!r}")

    # 5. Save & reload
    print("\n--- Save / Load ---")
    store.save("my_store")
    reloaded = VectorStore.load("my_store", embedding_dim=embedder.embedding_dim)
    print(f"Reloaded: {reloaded}")


if __name__ == "__main__":
    main()

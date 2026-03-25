# vector-test

Text vectorization / semantic search demo using HuggingFace sentence-transformers and FAISS.

## Project structure

```
vector-test/
├── embedder.py       # TextEmbedder — wraps sentence-transformers
├── vector_store.py   # VectorStore  — FAISS-backed in-memory store with save/load
├── main.py           # End-to-end demo (embed corpus → search → pairwise similarity)
└── requirements.txt
```

## Quick start

### 1. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run the demo

```powershell
python main.py
```

The first run downloads the `all-MiniLM-L6-v2` model (~90 MB) from HuggingFace automatically.

---

## Key classes

### `TextEmbedder` (`embedder.py`)

| Method | Description |
|---|---|
| `__init__(model_name, device)` | Load a sentence-transformers model |
| `embed(texts, normalize=True)` | Encode one string or a list → `np.ndarray` |
| `similarity(vec_a, vec_b)` | Cosine similarity between two vectors |

**Recommended models**

| Model | Dim | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Default — fast, lightweight |
| `all-mpnet-base-v2` | 768 | Higher quality, ~5× slower |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Tuned for Q&A retrieval |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ languages |

### `VectorStore` (`vector_store.py`)

| Method | Description |
|---|---|
| `add(vectors, texts, metadata)` | Insert embeddings + source texts |
| `search(query_vector, top_k)` | Return top-k `SearchResult` objects |
| `save(path)` | Persist to `<path>.faiss` + `<path>.npz` |
| `VectorStore.load(path, dim)` | Reload from disk |
| `reset()` | Clear all data |

---

## Usage in your own code

```python
from embedder import TextEmbedder
from vector_store import VectorStore

embedder = TextEmbedder()                       # downloads model once
store    = VectorStore(embedder.embedding_dim)

# Index documents
docs = ["Document one.", "Document two.", "Document three."]
store.add(embedder.embed(docs), texts=docs)

# Query
results = store.search(embedder.embed("my question"), top_k=2)
for r in results:
    print(r.score, r.text)
```

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Pre-trained embedding models |
| `transformers` | HuggingFace model hub |
| `torch` | PyTorch backend |
| `faiss-cpu` | Efficient similarity search |
| `numpy` | Array operations |
| `scikit-learn` | Utilities (optional) |
| `tqdm` | Progress bars |

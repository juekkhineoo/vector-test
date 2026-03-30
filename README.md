# vector-test

Semantic search over real datasets using HuggingFace sentence-transformers and FAISS.  
Supports CSV, JSON/JSONL, and plain-text corpora with an interactive query REPL.

## Project structure

```
vector-test/
├── embedder.py        # TextEmbedder   — wraps sentence-transformers
├── vector_store.py    # VectorStore    — FAISS-backed store with save/load
├── dataset_loader.py  # load_dataset() — CSV / JSON / JSONL / TXT loader
├── main.py            # CLI entry-point + interactive search loop
├── requirements.txt
└── TA13.csv           # Example dataset (hotel reviews, 80 k rows)
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

### 3. Run

```powershell
python main.py                              # default: TA13.csv, top-5 results
```

The first run downloads `all-MiniLM-L6-v2` (~90 MB) from HuggingFace and builds the
FAISS index. The index is cached to disk; subsequent runs load it instantly.

---

## CLI options

```
python main.py [OPTIONS]

Options:
  --dataset PATH        Dataset file to index (.csv, .json, .jsonl, .txt)
                        Default: TA13.csv
  --text-column NAME    Column / key containing the main text (auto-detected)
  --model NAME          HuggingFace sentence-transformers model
                        Default: all-MiniLM-L6-v2
  --top-k INT           Results returned per query  (default: 5)
  --store PATH          Prefix for the saved index files
                        Default: <dataset_stem>_<model_name> (e.g. TA13_all_MiniLM_L6_v2)
  --no-cache            Force rebuild of the vector index
  --batch-size INT      Embedding batch size  (default: 64)
```

**Examples**

```powershell
python main.py --dataset reviews.csv --top-k 3
python main.py --dataset articles.json --text-column body
python main.py --dataset notes.txt --model all-mpnet-base-v2
python main.py --no-cache                     # rebuild index from scratch
python main.py --store my_index               # custom index path prefix
```

**Interactive REPL**

After the index is ready you are dropped into a search prompt. Type your query and press Enter.  
To exit, type `quit`, `exit`, or `q` (or press `Ctrl+C`).

---

## Dataset loader

`dataset_loader.load_dataset(path)` auto-detects the file format:

| Extension | Behaviour |
|---|---|
| `.csv` | Reads with `csv.DictReader`; text column auto-detected from common names (`review_text`, `text`, `content`, `body`, `description`, `comment`, `review`, `passage`, `sentence`, `document`, `abstract`, `title`, `question`, `answer`) or falls back to the last column |
| `.json` | Handles list-of-objects, list-of-strings, single object |
| `.jsonl` | One JSON object per line |
| anything else | Each non-empty line → one document |

All non-text fields are preserved as **metadata** and displayed alongside search results.

---

## Key classes

### `TextEmbedder` (`embedder.py`)

| Method | Description |
|---|---|
| `__init__(model_name, device)` | Load a sentence-transformers model |
| `embed(texts, batch_size=64, normalize=True, show_progress=False)` | Encode one string or a list → `np.ndarray` |
| `similarity(vec_a, vec_b)` | Cosine similarity between two normalized vectors |

**Recommended models**

| Model | Dim | Notes |
|---|---|---|
| `all-MiniLM-L6-v2` | 384 | Default — fast, lightweight |
| `all-mpnet-base-v2` | 768 | Higher quality, ~5× slower |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Tuned for Q&A retrieval |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ languages |

### `SearchResult` (dataclass)

| Field | Type | Description |
|---|---|---|
| `index` | `int` | Position in the store |
| `score` | `float` | Similarity score (cosine: higher is better) |
| `text` | `str` | Source text |
| `metadata` | `dict` | Arbitrary per-item metadata |

### `VectorStore` (`vector_store.py`)

| Method | Description |
|---|---|
| `__init__(embedding_dim, metric='cosine')` | Create store; `metric` is `'cosine'` or `'l2'` |
| `add(vectors, texts, metadata=None)` | Insert embeddings + source texts |
| `search(query_vector, top_k)` | Return top-k `SearchResult` objects |
| `save(path)` | Persist to `<path>.faiss` + `<path>.npz` |
| `VectorStore.load(path, embedding_dim, metric='cosine')` | Reload from disk |
| `reset()` | Clear all data |
| `__len__()` | Number of indexed documents |

---

## Usage in your own code

```python
from embedder import TextEmbedder
from vector_store import VectorStore
from dataset_loader import load_dataset

embedder = TextEmbedder()                        # downloads model once
store    = VectorStore(embedder.embedding_dim)

# Load and index a dataset
records = load_dataset("my_data.csv")            # or .json / .txt
texts    = [r["text"]     for r in records]
metadata = [r["metadata"] for r in records]
store.add(embedder.embed(texts, show_progress=True), texts=texts, metadata=metadata)
store.save("my_index")

# Reload later  (metric must match what was used when saving)
store = VectorStore.load("my_index", embedding_dim=embedder.embedding_dim, metric="cosine")

# Query
results = store.search(embedder.embed("my question"), top_k=5)
for r in results:
    print(f"{r.score:.4f}  {r.text[:120]}")
```

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Pre-trained HuggingFace embedding models |
| `transformers` | HuggingFace model hub |
| `torch` | PyTorch backend |
| `faiss-cpu` | Efficient similarity search |
| `numpy` | Array operations |
| `scikit-learn` | Utilities (optional) |
| `tqdm` | Progress bars |

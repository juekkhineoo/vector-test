"""
Utilities for loading datasets from CSV, JSON, or plain text files.

Each loader returns a list of records:
    [{"text": str, "metadata": dict}, ...]
"""

import csv
import json
import os
from pathlib import Path

# Ordered list of common column/key names that likely hold the main text.
_TEXT_COLUMN_CANDIDATES = [
    "review_text", "text", "content", "body", "description",
    "comment", "review", "passage", "sentence", "document", "abstract",
    "title", "question", "answer",
]


def _pick_text_column(fieldnames: list[str]) -> str:
    """Return the best text column from a list of field names."""
    lower_map = {f.lower(): f for f in fieldnames}
    for candidate in _TEXT_COLUMN_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    # Fall back to the last field
    return fieldnames[-1]


def load_csv(path: str, text_column: str | None = None) -> list[dict]:
    """
    Load a CSV file and return records.

    Args:
        path:        Path to the CSV file.
        text_column: Column name containing the main text.
                     Auto-detected from common names if omitted.

    Returns:
        List of {"text": str, "metadata": dict} records.
    """
    records = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields: list[str] = list(reader.fieldnames or [])
        if not fields:
            return records

        col = text_column or _pick_text_column(fields)
        meta_cols = [f for f in fields if f != col]
        source = os.path.basename(path)

        for row_idx, row in enumerate(reader):
            text = (row.get(col) or "").strip()
            if not text:
                continue
            metadata: dict = {k: row[k] for k in meta_cols if row.get(k)}
            metadata["source"] = source
            metadata["row"] = row_idx
            records.append({"text": text, "metadata": metadata})
    return records


def load_json(path: str, text_key: str | None = None) -> list[dict]:
    """
    Load a JSON or JSONL file and return records.

    Handles:
      - List of objects: [{"text": ...}, ...]
      - List of strings: ["...", "..."]
      - Single object:   {"text": ...}
      - JSONL:           one JSON object per line

    Args:
        path:     Path to the JSON/JSONL file.
        text_key: Key containing the main text. Auto-detected if omitted.

    Returns:
        List of {"text": str, "metadata": dict} records.
    """
    source = os.path.basename(path)
    ext = Path(path).suffix.lower()

    # JSONL: one JSON object per line
    if ext == ".jsonl":
        items = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    else:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        items = [data] if isinstance(data, dict) else data

    records = []
    for row_idx, item in enumerate(items):
        if isinstance(item, str):
            text = item.strip()
            metadata: dict = {}
        elif isinstance(item, dict):
            col = text_key or _pick_text_column(list(item.keys()))
            text = str(item.get(col, "")).strip()
            metadata = {k: v for k, v in item.items() if k != col}
        else:
            continue

        if not text:
            continue
        metadata["source"] = source
        metadata["row"] = row_idx
        records.append({"text": text, "metadata": metadata})
    return records


def load_text(path: str) -> list[dict]:
    """
    Load a plain text file. Each non-empty line becomes one document.

    Returns:
        List of {"text": str, "metadata": dict} records.
    """
    source = os.path.basename(path)
    records = []
    with open(path, encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            text = raw.strip()
            if text:
                records.append({
                    "text": text,
                    "metadata": {"source": source, "line": line_no},
                })
    return records


def load_dataset(path: str, text_column: str | None = None) -> list[dict]:
    """
    Auto-detect file type and load the dataset.

    Supported formats:
        .csv              → load_csv
        .json / .jsonl    → load_json
        anything else     → load_text (line-per-document)

    Args:
        path:        Path to the dataset file.
        text_column: Column/key that holds the main text (auto-detected if omitted).

    Returns:
        List of {"text": str, "metadata": dict} records.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset file not found: {path!r}")

    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return load_csv(path, text_column=text_column)
    elif ext in (".json", ".jsonl"):
        return load_json(path, text_key=text_column)
    else:
        return load_text(path)

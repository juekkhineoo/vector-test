"""
PostgreSQL + pgvector-backed vector store.

Requires:
  - PostgreSQL with the pgvector extension installed.
  - psycopg2-binary and pgvector Python packages.

Connection string format:
    postgresql://user:password@host:port/dbname
"""

from __future__ import annotations

import json

import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    index: int
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """
    Vector store backed by PostgreSQL + pgvector.

    Documents are persisted in a 'documents' table with a vector column.
    Similarity search uses cosine distance via the <=> operator.
    """

    def __init__(self, conn_str: str, embedding_dim: int):
        """
        Args:
            conn_str:      PostgreSQL connection string.
            embedding_dim: Dimensionality of the embedding vectors.
        """
        self.embedding_dim = embedding_dim
        self.conn = psycopg2.connect(conn_str)
        self.is_new: bool = self._init_table()  # CREATE EXTENSION + table; True = first run
        register_vector(self.conn)              # register vector type OID with psycopg2

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_table(self) -> bool:
        """
        Create the pgvector extension and documents table if absent.

        Returns:
            True  — table was just created (first run).
            False — table already existed.
        """
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Check whether the table already exists before creating it
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = 'public'
                      AND table_name   = 'documents'
                );
            """)
            already_existed: bool = cur.fetchone()[0]

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id        SERIAL PRIMARY KEY,
                    text      TEXT    NOT NULL,
                    metadata  JSONB,
                    embedding vector({self.embedding_dim})
                );
            """)
            # HNSW index for fast approximate cosine search.
            # Works at any table size (unlike IVFFlat which needs a minimum row count).
            cur.execute("""
                CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx
                ON documents USING hnsw (embedding vector_cosine_ops);
            """)
        self.conn.commit()
        return not already_existed

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------

    def add(
        self,
        vectors: np.ndarray,
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """
        Bulk-insert vectors and their source texts into the database.

        Args:
            vectors:  float32 array of shape (N, embedding_dim).
            texts:    List of N source strings.
            metadata: Optional list of N dicts with arbitrary per-item metadata.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        if metadata is None:
            metadata = [{} for _ in texts]

        rows = [
            (text, json.dumps(meta), vec.tolist())
            for text, meta, vec in zip(texts, metadata, vectors)
        ]
        with self.conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO documents (text, metadata, embedding) VALUES %s",
                rows,
                template="(%s, %s::jsonb, %s::vector)",
            )
        self.conn.commit()

    def reset(self) -> None:
        """Delete all rows and reset the ID sequence."""
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE documents RESTART IDENTITY;")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Return the top-k most similar documents using cosine similarity.

        Args:
            query_vector: 1-D float32 array of length embedding_dim.
            top_k:        Number of results to return.

        Returns:
            List of SearchResult sorted by score descending.
        """
        query = np.asarray(query_vector, dtype=np.float32).tolist()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, text, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query, query, top_k),
            )
            rows = cur.fetchall()

        results = []
        for row_id, text, meta, score in rows:
            if isinstance(meta, str):
                meta = json.loads(meta)
            results.append(
                SearchResult(
                    index=int(row_id),
                    score=float(score),
                    text=text,
                    metadata=meta or {},
                )
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents;")
            return cur.fetchone()[0]

    def __repr__(self) -> str:
        return (
            f"VectorStore(items={len(self)}, dim={self.embedding_dim}, backend=postgresql)"
        )

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

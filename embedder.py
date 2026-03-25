"""
Text embedding using HuggingFace sentence-transformers models.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Union


class TextEmbedder:
    """
    Wrapper around HuggingFace sentence-transformers for generating text embeddings.

    Default model: all-MiniLM-L6-v2  (fast, lightweight, good quality)
    Other good options:
      - all-mpnet-base-v2       (higher quality, slower)
      - multi-qa-MiniLM-L6-cos-v1 (tuned for Q&A retrieval)
      - paraphrase-multilingual-MiniLM-L12-v2 (multilingual)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        print(f"Loading model '{model_name}' on {device} ...")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed(
        self,
        texts: Union[str, list[str]],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed one or more texts into vectors.

        Args:
            texts:         A single string or a list of strings.
            batch_size:    Number of texts processed per forward pass.
            normalize:     L2-normalize output vectors (recommended for cosine similarity).
            show_progress: Show a tqdm progress bar.

        Returns:
            np.ndarray of shape (N, embedding_dim) where N = len(texts).
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        return vectors[0] if single else vectors

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two normalized vectors."""
        return float(np.dot(vec_a, vec_b))

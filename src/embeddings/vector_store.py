"""FAISS-based vector store for efficient similarity search over paper embeddings."""

from __future__ import annotations
import logging
import pickle
import numpy as np
import faiss
from pathlib import Path
from typing import Optional

from src.models.paper import Paper, PaperCollection
from src.config import FAISS_DIR, EMBEDDING_DIM

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Manages a FAISS index for paper embeddings with metadata mapping."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        self.index: Optional[faiss.Index] = None
        self.paper_ids: list[str] = []
        self.id_to_idx: dict[str, int] = {}

    def build(self, papers: list[Paper], embeddings: np.ndarray) -> None:
        """Build a FAISS index from paper embeddings.

        Uses IndexFlatIP (inner product) since embeddings are L2-normalized,
        making inner product equivalent to cosine similarity.
        """
        assert len(papers) == len(embeddings), "Papers and embeddings count mismatch"

        # Auto-detect dimension from actual embeddings
        self.dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings.astype(np.float32))

        self.paper_ids = [p.paper_id for p in papers]
        self.id_to_idx = {pid: i for i, pid in enumerate(self.paper_ids)}

        logger.info(f"Built FAISS index with {self.index.ntotal} vectors (dim={self.dim})")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search for nearest neighbors of a query embedding.

        Args:
            query_embedding: Query vector of shape (1, dim) or (dim,).
            top_k: Number of results to return.

        Returns:
            List of (paper_id, similarity_score) tuples sorted by descending similarity.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        qvec = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(qvec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.paper_ids):
                continue
            results.append((self.paper_ids[idx], float(score)))

        return results

    def find_similar(
        self,
        paper_id: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find papers similar to a given paper."""
        if paper_id not in self.id_to_idx:
            return []

        idx = self.id_to_idx[paper_id]
        vec = self.index.reconstruct(idx).reshape(1, -1)
        all_results = self.search(vec, top_k=top_k + 1)
        return [(pid, s) for pid, s in all_results if pid != paper_id][:top_k]

    def save(self, name: str) -> None:
        """Persist the FAISS index and metadata to disk."""
        index_path = FAISS_DIR / f"{name}.index"
        meta_path = FAISS_DIR / f"{name}_meta.pkl"

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"paper_ids": self.paper_ids, "dim": self.dim}, f)

        logger.info(f"Saved FAISS index to {index_path}")

    def load(self, name: str) -> bool:
        """Load a FAISS index and metadata from disk."""
        index_path = FAISS_DIR / f"{name}.index"
        meta_path = FAISS_DIR / f"{name}_meta.pkl"

        if not index_path.exists() or not meta_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self.paper_ids = meta["paper_ids"]
        self.dim = meta["dim"]
        self.id_to_idx = {pid: i for i, pid in enumerate(self.paper_ids)}

        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors from {index_path}")
        return True

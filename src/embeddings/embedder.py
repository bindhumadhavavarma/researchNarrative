"""SPECTER-based embedding pipeline for scientific papers.

Uses the allenai/specter2 model (or fallback to all-MiniLM-L6-v2)
to generate document-level embeddings from title + abstract.
"""

from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from src.models.paper import Paper, PaperCollection
from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDINGS_DIR,
)

logger = logging.getLogger(__name__)

# Lazy-loaded model to avoid slow imports on startup
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        try:
            logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            fallback = "all-MiniLM-L6-v2"
            logger.warning(
                f"Failed to load {EMBEDDING_MODEL}: {e}. "
                f"Falling back to {fallback}"
            )
            _model = SentenceTransformer(fallback)
    return _model


class PaperEmbedder:
    """Generates and manages embeddings for paper collections."""

    def __init__(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.batch_size = batch_size

    def embed_collection(
        self,
        collection: PaperCollection,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for all papers in a collection.

        Args:
            collection: PaperCollection to embed.
            force_recompute: If True, recompute even if cached.

        Returns:
            numpy array of shape (n_papers, embedding_dim).
        """
        safe_topic = collection.topic.replace(" ", "_").replace("/", "_")[:50]
        cache_path = EMBEDDINGS_DIR / f"{safe_topic}_embeddings.npy"
        ids_path = EMBEDDINGS_DIR / f"{safe_topic}_ids.npy"

        papers = collection.get_papers()
        paper_ids = [p.paper_id for p in papers]

        if not force_recompute and cache_path.exists() and ids_path.exists():
            cached_embeddings = np.load(cache_path)
            cached_ids = np.load(ids_path, allow_pickle=True).tolist()
            if cached_ids == paper_ids and len(cached_embeddings) == len(papers):
                logger.info(f"Loaded cached embeddings from {cache_path}")
                for i, paper in enumerate(papers):
                    paper.embedding = cached_embeddings[i].tolist()
                return cached_embeddings

        logger.info(f"Computing embeddings for {len(papers)} papers...")
        texts = [p.embedding_text() for p in papers]
        model = _get_model()

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        for i, paper in enumerate(papers):
            paper.embedding = embeddings[i].tolist()

        np.save(cache_path, embeddings)
        np.save(ids_path, np.array(paper_ids, dtype=object))
        logger.info(f"Saved embeddings to {cache_path} — shape: {embeddings.shape}")

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string for similarity search."""
        model = _get_model()
        embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )
        return np.array(embedding, dtype=np.float32)

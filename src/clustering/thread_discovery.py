"""HDBSCAN-based clustering for research thread discovery.

Identifies distinct research threads within a paper collection by:
1. Reducing embedding dimensionality with UMAP
2. Clustering with HDBSCAN
3. Labeling clusters with LLM assistance
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Optional
from collections import Counter

from src.models.paper import Paper, PaperCollection
from src.config import (
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    UMAP_N_NEIGHBORS,
    UMAP_N_COMPONENTS,
    UMAP_METRIC,
    HAS_LLM,
)
from src.utils.llm import get_llm_client, get_model_name

logger = logging.getLogger(__name__)


class ThreadDiscovery:
    """Discovers research threads via clustering."""

    def __init__(
        self,
        min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples: int = HDBSCAN_MIN_SAMPLES,
        umap_n_neighbors: int = UMAP_N_NEIGHBORS,
        umap_n_components: int = UMAP_N_COMPONENTS,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.reduced_embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.umap_2d: Optional[np.ndarray] = None  # for visualization

    def cluster(
        self,
        papers: list[Paper],
        embeddings: np.ndarray,
    ) -> dict[int, list[Paper]]:
        """Cluster papers into research threads.

        Args:
            papers: List of Paper objects.
            embeddings: Array of shape (n_papers, embedding_dim).

        Returns:
            Dictionary mapping cluster_id -> list of papers.
            Cluster -1 contains noise/unclustered papers.
        """
        import umap
        import hdbscan

        n_papers = len(papers)
        logger.info(f"Clustering {n_papers} papers...")

        # Dimensionality reduction with UMAP
        n_components = min(self.umap_n_components, n_papers - 2)
        n_neighbors = min(self.umap_n_neighbors, n_papers - 1)

        if n_papers < self.min_cluster_size + 2:
            logger.warning(f"Too few papers ({n_papers}) for meaningful clustering")
            for p in papers:
                p.cluster_id = 0
                p.cluster_label = "All Papers"
            return {0: papers}

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric=UMAP_METRIC,
            random_state=42,
        )
        self.reduced_embeddings = reducer.fit_transform(embeddings)

        # Also compute 2D projection for visualization
        reducer_2d = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            metric=UMAP_METRIC,
            random_state=42,
        )
        self.umap_2d = reducer_2d.fit_transform(embeddings)

        # HDBSCAN clustering
        min_cluster = min(self.min_cluster_size, max(2, n_papers // 5))
        min_samp = min(self.min_samples, min_cluster)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster,
            min_samples=min_samp,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        self.labels = clusterer.fit_predict(self.reduced_embeddings)

        clusters: dict[int, list[Paper]] = {}
        for paper, label in zip(papers, self.labels):
            label = int(label)
            paper.cluster_id = label
            clusters.setdefault(label, []).append(paper)

        n_clusters = len([c for c in clusters if c != -1])
        noise_count = len(clusters.get(-1, []))
        logger.info(
            f"Found {n_clusters} clusters, {noise_count} noise papers"
        )

        return clusters

    def label_clusters(
        self,
        clusters: dict[int, list[Paper]],
    ) -> dict[int, str]:
        """Generate human-readable labels for each cluster.

        Uses LLM if available, otherwise falls back to keyword extraction.
        """
        labels = {}

        for cluster_id, papers in clusters.items():
            if cluster_id == -1:
                labels[-1] = "Unclustered"
                for p in papers:
                    p.cluster_label = "Unclustered"
                continue

            if HAS_LLM:
                label = self._llm_label(cluster_id, papers)
            else:
                label = self._keyword_label(papers)

            labels[cluster_id] = label
            for p in papers:
                p.cluster_label = label

        logger.info(f"Cluster labels: {labels}")
        return labels

    def _llm_label(self, cluster_id: int, papers: list[Paper]) -> str:
        """Use LLM to generate a concise cluster label."""
        try:
            client = get_llm_client()
            model = get_model_name()

            sample = sorted(papers, key=lambda p: p.citation_count, reverse=True)[:8]
            titles = "\n".join(f"- {p.title} ({p.year})" for p in sample)

            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at labeling research clusters. "
                            "Given paper titles from one cluster, produce a short "
                            "(3-6 word) label that captures the research thread. "
                            "Reply with ONLY the label, nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Cluster {cluster_id} papers:\n{titles}",
                    },
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM labeling failed: {e}")
            return self._keyword_label(papers)

    def _keyword_label(self, papers: list[Paper]) -> str:
        """Fallback: extract frequent meaningful words from titles."""
        stop_words = {
            "a", "an", "the", "of", "for", "in", "on", "to", "and", "with",
            "is", "are", "by", "from", "at", "as", "or", "using", "via",
            "based", "towards", "through", "its", "their", "this", "that",
        }
        word_counts: Counter = Counter()
        for paper in papers:
            words = paper.title.lower().split()
            for w in words:
                cleaned = w.strip(".,;:!?()[]\"'")
                if cleaned and len(cleaned) > 2 and cleaned not in stop_words:
                    word_counts[cleaned] += 1

        top_words = [w for w, _ in word_counts.most_common(4)]
        return " ".join(top_words).title() if top_words else f"Cluster"

    def get_cluster_stats(
        self, clusters: dict[int, list[Paper]]
    ) -> list[dict]:
        """Compute summary statistics for each cluster."""
        stats = []
        for cluster_id, papers in sorted(clusters.items()):
            if cluster_id == -1:
                continue
            years = [p.year for p in papers if p.year]
            citations = [p.citation_count for p in papers]
            stats.append({
                "cluster_id": cluster_id,
                "label": papers[0].cluster_label if papers else "",
                "size": len(papers),
                "year_range": f"{min(years)}-{max(years)}" if years else "N/A",
                "avg_citations": round(np.mean(citations), 1) if citations else 0,
                "total_citations": sum(citations),
                "top_paper": max(papers, key=lambda p: p.citation_count).title if papers else "",
            })
        return stats

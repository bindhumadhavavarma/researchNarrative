"""End-to-end pipeline orchestrator for ResearchNarrative.

Chains: topic query -> paper retrieval -> embedding -> clustering -> narrative.
"""

from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Callable
from pathlib import Path

from src.models.paper import PaperCollection
from src.api.ingestion import PaperIngestionPipeline
from src.embeddings.embedder import PaperEmbedder
from src.embeddings.vector_store import FAISSVectorStore
from src.clustering.thread_discovery import ThreadDiscovery
from src.narrative.generator import NarrativeGenerator
from src.config import PAPERS_DIR

logger = logging.getLogger(__name__)


class ResearchNarrativePipeline:
    """Orchestrates the full ResearchNarrative workflow."""

    def __init__(self):
        self.ingestion = PaperIngestionPipeline()
        self.embedder = PaperEmbedder()
        self.vector_store = FAISSVectorStore()
        self.thread_discovery = ThreadDiscovery()
        self.narrative_gen = NarrativeGenerator()

        # State
        self.collection: Optional[PaperCollection] = None
        self.embeddings: Optional[np.ndarray] = None
        self.clusters: Optional[dict] = None
        self.cluster_labels: Optional[dict] = None
        self.narrative: Optional[str] = None

    def run(
        self,
        topic: str,
        max_papers: int = 200,
        sources: Optional[list[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        min_citations: int = 0,
        enrich_citations: bool = True,
        skip_ingestion: bool = False,
        progress_callback: Optional[Callable[[str, str], None]] = None,
    ) -> dict:
        """Execute the full pipeline.

        Args:
            progress_callback: Optional function(step_label, detail_message) to report progress.
        """
        def _report(step: str, msg: str):
            logger.info(f"[{step}] {msg}")
            if progress_callback:
                progress_callback(step, msg)

        _report("init", f"Starting pipeline for: '{topic}'")

        # Reset state for new run
        self.collection = None
        self.embeddings = None
        self.clusters = None
        self.cluster_labels = None
        self.narrative = None

        # Step 1: Paper Ingestion
        _report("ingestion", "Collecting papers from APIs...")
        if skip_ingestion:
            self.collection = self._load_cached(topic)
        if self.collection is None:
            def _ingest_progress(msg):
                _report("ingestion", msg)

            self.collection = self.ingestion.ingest(
                topic=topic,
                max_papers=max_papers,
                sources=sources,
                start_year=start_year,
                end_year=end_year,
                min_citations=min_citations,
                enrich_citations=enrich_citations,
                progress_callback=_ingest_progress,
            )

        papers = self.collection.get_papers()
        if not papers:
            return {"error": "No papers found for this topic. Try a different query or broader year range."}

        _report("ingestion", f"Step 1 complete: {len(papers)} papers collected")

        # Step 2: Embedding
        _report("embedding", f"Generating SPECTER embeddings for {len(papers)} papers...")
        self.embeddings = self.embedder.embed_collection(self.collection)
        _report("embedding", f"Embeddings ready — shape: {self.embeddings.shape}")

        # Step 3: FAISS Index
        _report("indexing", "Building FAISS similarity index...")
        self.vector_store.build(papers, self.embeddings)
        safe_topic = topic.replace(" ", "_").replace("/", "_")[:50]
        self.vector_store.save(safe_topic)
        _report("indexing", f"FAISS index built with {len(papers)} vectors")

        # Step 4: Clustering
        _report("clustering", "Discovering research threads (UMAP + HDBSCAN)...")
        self.clusters = self.thread_discovery.cluster(papers, self.embeddings)
        n_clusters = len([c for c in self.clusters if c != -1])
        _report("clustering", f"Found {n_clusters} research threads")

        _report("clustering", "Labeling clusters...")
        self.cluster_labels = self.thread_discovery.label_clusters(self.clusters)
        cluster_stats = self.thread_discovery.get_cluster_stats(self.clusters)
        labels_str = ", ".join(f'"{v}"' for k, v in self.cluster_labels.items() if k != -1)
        _report("clustering", f"Threads: {labels_str}")

        # Step 5: Narrative Generation
        _report("narrative", "Generating research narrative...")
        self.narrative = self.narrative_gen.generate(
            topic=topic,
            clusters=self.clusters,
            cluster_labels=self.cluster_labels,
        )
        _report("narrative", f"Narrative generated ({len(self.narrative)} characters)")

        _report("done", "Pipeline complete!")

        return {
            "topic": topic,
            "papers": papers,
            "embeddings": self.embeddings,
            "clusters": self.clusters,
            "cluster_labels": self.cluster_labels,
            "cluster_stats": cluster_stats,
            "narrative": self.narrative,
            "vector_store": self.vector_store,
            "umap_2d": self.thread_discovery.umap_2d,
        }

    def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search for papers similar to a text query."""
        query_emb = self.embedder.embed_query(query)
        return self.vector_store.search(query_emb, top_k=top_k)

    def _load_cached(self, topic: str) -> Optional[PaperCollection]:
        safe_topic = topic.replace(" ", "_").replace("/", "_")[:50]
        path = PAPERS_DIR / f"{safe_topic}.json"
        if path.exists():
            logger.info(f"Loading cached papers from {path}")
            return PaperCollection.load(path)
        return None

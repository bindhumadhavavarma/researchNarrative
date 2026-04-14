"""End-to-end pipeline orchestrator for ResearchNarrative.

Chains: topic query -> paper retrieval -> embedding -> clustering
        -> citation graph analysis -> narrative.
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
from src.citation.graph import CitationGraph
from src.citation.influence import InfluenceScorer
from src.citation.competition import CompetitionDetector
from src.evaluation.metrics import PipelineEvaluator
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
        self.evaluator = PipelineEvaluator()

        # State
        self.collection: Optional[PaperCollection] = None
        self.embeddings: Optional[np.ndarray] = None
        self.clusters: Optional[dict] = None
        self.cluster_labels: Optional[dict] = None
        self.narrative: Optional[str] = None
        self.citation_graph: Optional[CitationGraph] = None
        self.influence_scores: Optional[dict] = None
        self.competition_analysis: Optional[dict] = None
        self.evaluation_results: Optional[dict] = None

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
        self.citation_graph = None
        self.influence_scores = None
        self.competition_analysis = None
        self.evaluation_results = None

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

        # Step 5: Citation Graph Analysis
        _report("citation_graph", "Building citation graph...")
        self.citation_graph = CitationGraph()
        self.citation_graph.build(papers)
        n_edges = self.citation_graph.graph.number_of_edges()
        _report("citation_graph", f"Citation graph: {len(papers)} nodes, {n_edges} edges")

        _report("citation_graph", "Computing influence scores (PageRank, HITS, bridge, burst)...")
        scorer = InfluenceScorer(self.citation_graph)
        self.influence_scores = scorer.compute_all(papers)
        top_influential = scorer.get_top_influential(5)
        if top_influential:
            top_names = ", ".join(
                f'"{self.citation_graph.get_paper(pid).title[:40]}..."'
                for pid, _ in top_influential
                if self.citation_graph.get_paper(pid)
            )
            _report("citation_graph", f"Top influential: {top_names}")

        paradigm_shifters = scorer.get_paradigm_shifters(papers)
        _report("citation_graph", f"Identified {len(paradigm_shifters)} potential paradigm-shifting papers")

        _report("citation_graph", "Analyzing competition and dominance between threads...")
        detector = CompetitionDetector(self.citation_graph)
        self.competition_analysis = detector.analyze(papers, self.clusters, self.cluster_labels)
        n_competing = len(self.competition_analysis.get("competition_pairs", []))
        n_complementary = len(self.competition_analysis.get("complementary_pairs", []))
        _report("citation_graph", f"Found {n_competing} competing pairs, {n_complementary} complementary pairs")

        # Step 6: Narrative Generation
        _report("narrative", "Generating research narrative...")

        def _narrative_progress(msg):
            _report("narrative", msg)

        self.narrative = self.narrative_gen.generate(
            topic=topic,
            clusters=self.clusters,
            cluster_labels=self.cluster_labels,
            influence_scores=self.influence_scores,
            competition_analysis=self.competition_analysis,
            citation_graph=self.citation_graph,
            progress_callback=_narrative_progress,
        )
        _report("narrative", f"Narrative generated ({len(self.narrative)} characters)")

        # Step 7: Evaluation
        _report("evaluation", "Running automated evaluation metrics...")
        self.evaluation_results = self.evaluator.evaluate_all(
            papers=papers,
            embeddings=self.embeddings,
            clusters=self.clusters,
            cluster_labels=self.cluster_labels,
            narrative=self.narrative,
            citation_verification=self.narrative_gen.verification_result,
            influence_scores=self.influence_scores,
            citation_graph=self.citation_graph,
        )
        grade = self.evaluator.get_grade()
        overall = self.evaluation_results.get("overall", {}).get("score", 0)
        _report("evaluation", f"Evaluation complete — Grade: {grade} (score: {overall:.3f})")

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
            "citation_graph": self.citation_graph,
            "influence_scores": self.influence_scores,
            "competition_analysis": self.competition_analysis,
            "citation_verification": self.narrative_gen.verification_result,
            "thread_narratives": self.narrative_gen.thread_narratives,
            "evaluation": self.evaluation_results,
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

"""Automated evaluation metrics for ResearchNarrative pipeline output.

Measures clustering quality, citation coverage, narrative quality,
and retrieval effectiveness — providing quantitative evidence of system
performance without requiring manual assessment.
"""

from __future__ import annotations
import logging
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

from src.models.paper import Paper

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Computes evaluation metrics across the full pipeline."""

    def __init__(self):
        self.metrics: dict[str, dict] = {}

    def evaluate_all(
        self,
        papers: list[Paper],
        embeddings: np.ndarray,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
        narrative: str,
        citation_verification: Optional[dict] = None,
        influence_scores: Optional[dict] = None,
        citation_graph=None,
    ) -> dict[str, dict]:
        """Run all evaluation metrics and return structured results."""
        self.metrics = {}
        self.metrics["retrieval"] = self._eval_retrieval(papers)
        self.metrics["clustering"] = self._eval_clustering(papers, embeddings, clusters)
        self.metrics["citation_graph"] = self._eval_citation_graph(papers, citation_graph)
        self.metrics["narrative"] = self._eval_narrative(
            narrative, papers, citation_verification
        )
        self.metrics["overall"] = self._compute_overall_score()
        return self.metrics

    def _eval_retrieval(self, papers: list[Paper]) -> dict:
        """Evaluate retrieval quality: source diversity, abstract coverage, metadata completeness."""
        total = len(papers)
        if total == 0:
            return {"score": 0, "details": {}}

        source_counts = Counter(p.source for p in papers)
        source_diversity = len(source_counts) / max(len(source_counts), 1)

        has_abstract = sum(1 for p in papers if p.abstract and len(p.abstract) > 50)
        abstract_coverage = has_abstract / total

        has_year = sum(1 for p in papers if p.year)
        has_authors = sum(1 for p in papers if p.authors)
        has_citations = sum(1 for p in papers if p.citation_count > 0)
        metadata_completeness = (
            (has_year + has_authors + has_citations) / (3 * total)
        )

        years = [p.year for p in papers if p.year]
        year_span = max(years) - min(years) if len(years) >= 2 else 0

        unique_authors = set()
        for p in papers:
            for a in p.authors:
                unique_authors.add(a.name.lower())

        score = (
            0.30 * abstract_coverage
            + 0.25 * metadata_completeness
            + 0.25 * min(source_diversity, 1.0)
            + 0.20 * min(year_span / 15, 1.0)
        )

        return {
            "score": round(score, 3),
            "details": {
                "total_papers": total,
                "source_distribution": dict(source_counts),
                "abstract_coverage": round(abstract_coverage, 3),
                "metadata_completeness": round(metadata_completeness, 3),
                "year_span": year_span,
                "unique_authors": len(unique_authors),
            },
        }

    def _eval_clustering(
        self,
        papers: list[Paper],
        embeddings: np.ndarray,
        clusters: dict[int, list[Paper]],
    ) -> dict:
        """Evaluate clustering quality using silhouette score and other metrics."""
        real_clusters = {cid: ps for cid, ps in clusters.items() if cid != -1}
        n_clusters = len(real_clusters)
        total = len(papers)
        noise_papers = len(clusters.get(-1, []))
        noise_ratio = noise_papers / total if total > 0 else 0

        labels = np.array([p.cluster_id for p in papers])
        non_noise_mask = labels != -1

        silhouette = -1.0
        if n_clusters >= 2 and np.sum(non_noise_mask) >= n_clusters + 1:
            try:
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(
                    embeddings[non_noise_mask],
                    labels[non_noise_mask],
                    metric="cosine",
                    sample_size=min(2000, int(np.sum(non_noise_mask))),
                )
            except Exception as e:
                logger.warning(f"Silhouette score failed: {e}")

        davies_bouldin = -1.0
        if n_clusters >= 2 and np.sum(non_noise_mask) >= n_clusters + 1:
            try:
                from sklearn.metrics import davies_bouldin_score
                davies_bouldin = davies_bouldin_score(
                    embeddings[non_noise_mask],
                    labels[non_noise_mask],
                )
            except Exception as e:
                logger.warning(f"Davies-Bouldin score failed: {e}")

        sizes = [len(ps) for ps in real_clusters.values()]
        size_std = float(np.std(sizes)) if sizes else 0
        size_cv = size_std / np.mean(sizes) if sizes and np.mean(sizes) > 0 else 0
        balance_score = max(0, 1.0 - size_cv)

        silhouette_norm = max(0, (silhouette + 1) / 2) if silhouette > -1 else 0.5
        noise_quality = max(0, 1.0 - noise_ratio * 2)

        score = (
            0.40 * silhouette_norm
            + 0.20 * balance_score
            + 0.20 * noise_quality
            + 0.20 * min(n_clusters / 5, 1.0)
        )

        return {
            "score": round(score, 3),
            "details": {
                "n_clusters": n_clusters,
                "silhouette_score": round(silhouette, 4) if silhouette > -1 else None,
                "davies_bouldin_index": round(davies_bouldin, 4) if davies_bouldin > -1 else None,
                "noise_ratio": round(noise_ratio, 3),
                "cluster_sizes": sizes,
                "balance_score": round(balance_score, 3),
            },
        }

    def _eval_citation_graph(self, papers: list[Paper], citation_graph) -> dict:
        """Evaluate citation graph connectivity and coverage."""
        total = len(papers)
        if total == 0 or citation_graph is None:
            return {"score": 0, "details": {}}

        g = citation_graph.graph
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()

        papers_with_refs = sum(1 for p in papers if p.references)
        papers_with_cited_by = sum(1 for p in papers if p.cited_by)
        enrichment_coverage = (papers_with_refs + papers_with_cited_by) / (2 * total)

        density = n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

        in_degrees = [g.in_degree(n) for n in g.nodes]
        connected_nodes = sum(1 for d in in_degrees if d > 0)
        connectivity = connected_nodes / n_nodes if n_nodes > 0 else 0

        avg_degree = np.mean(in_degrees) if in_degrees else 0

        score = (
            0.40 * enrichment_coverage
            + 0.30 * connectivity
            + 0.30 * min(avg_degree / 3, 1.0)
        )

        return {
            "score": round(score, 3),
            "details": {
                "nodes": n_nodes,
                "edges": n_edges,
                "density": round(density, 6),
                "enrichment_coverage": round(enrichment_coverage, 3),
                "connectivity": round(connectivity, 3),
                "avg_in_degree": round(avg_degree, 2),
            },
        }

    def _eval_narrative(
        self,
        narrative: str,
        papers: list[Paper],
        citation_verification: Optional[dict],
    ) -> dict:
        """Evaluate narrative quality: length, structure, citation density, accuracy."""
        if not narrative:
            return {"score": 0, "details": {}}

        word_count = len(narrative.split())
        sections = re.findall(r'^##\s+', narrative, re.MULTILINE)
        n_sections = len(sections)
        paragraphs = [p.strip() for p in narrative.split('\n\n') if p.strip() and not p.strip().startswith('#')]
        n_paragraphs = len(paragraphs)

        citation_pattern = re.compile(
            r'\[([A-Z][a-zA-Z\-\']+(?:\s+et\s+al\.)?),?\s*(\d{4})\]'
        )
        citations = citation_pattern.findall(narrative)
        n_citations = len(citations)
        unique_citations = len(set(citations))

        citation_density = n_citations / max(n_paragraphs, 1)

        total_papers = len(papers)
        papers_cited = set()
        for author_part, year_str in citations:
            surname = author_part.replace(" et al.", "").replace(" et al", "").strip().split()[-1].lower()
            year = int(year_str)
            for p in papers:
                if p.year == year and p.authors:
                    p_surname = p.authors[0].name.split()[-1].lower()
                    if p_surname == surname:
                        papers_cited.add(p.paper_id)
                        break
        coverage = len(papers_cited) / total_papers if total_papers > 0 else 0

        citation_accuracy = 1.0
        if citation_verification:
            citation_accuracy = citation_verification["stats"]["accuracy"]

        length_score = min(word_count / 1500, 1.0)
        structure_score = min(n_sections / 5, 1.0)

        score = (
            0.25 * citation_accuracy
            + 0.25 * min(citation_density / 2, 1.0)
            + 0.20 * min(coverage * 5, 1.0)
            + 0.15 * length_score
            + 0.15 * structure_score
        )

        return {
            "score": round(score, 3),
            "details": {
                "word_count": word_count,
                "n_sections": n_sections,
                "n_paragraphs": n_paragraphs,
                "n_citations": n_citations,
                "unique_citations": unique_citations,
                "citation_density": round(citation_density, 2),
                "paper_coverage": round(coverage, 3),
                "citation_accuracy": round(citation_accuracy, 3),
            },
        }

    def _compute_overall_score(self) -> dict:
        """Weighted combination of all sub-scores."""
        weights = {
            "retrieval": 0.20,
            "clustering": 0.25,
            "citation_graph": 0.20,
            "narrative": 0.35,
        }
        total = sum(
            self.metrics[k]["score"] * w
            for k, w in weights.items()
            if k in self.metrics
        )
        return {
            "score": round(total, 3),
            "weights": weights,
            "component_scores": {
                k: self.metrics[k]["score"]
                for k in weights
                if k in self.metrics
            },
        }

    def get_grade(self) -> str:
        """Return a letter grade based on the overall score."""
        score = self.metrics.get("overall", {}).get("score", 0)
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C"
        else:
            return "D"

    def get_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on evaluation results."""
        recs = []
        retrieval = self.metrics.get("retrieval", {})
        clustering = self.metrics.get("clustering", {})
        citation = self.metrics.get("citation_graph", {})
        narrative = self.metrics.get("narrative", {})

        r_details = retrieval.get("details", {})
        if r_details.get("abstract_coverage", 1) < 0.8:
            recs.append("Low abstract coverage — some papers may lack abstracts, which affects embedding quality.")
        if r_details.get("year_span", 0) < 5:
            recs.append("Narrow year range — consider broadening the time range for more comprehensive coverage.")

        c_details = clustering.get("details", {})
        sil = c_details.get("silhouette_score")
        if sil is not None and sil < 0.2:
            recs.append("Low silhouette score — clusters may not be well-separated. Try adjusting HDBSCAN parameters.")
        if c_details.get("noise_ratio", 0) > 0.3:
            recs.append("High noise ratio — many papers are unclustered. Lowering min_cluster_size may help.")

        cg_details = citation.get("details", {})
        if cg_details.get("connectivity", 1) < 0.3:
            recs.append("Low citation graph connectivity — enable citation enrichment for better graph coverage.")

        n_details = narrative.get("details", {})
        if n_details.get("citation_accuracy", 1) < 0.7:
            recs.append("Citation accuracy below 70% — narrative may reference papers outside the collection.")
        if n_details.get("citation_density", 0) < 1:
            recs.append("Low citation density — narrative paragraphs should cite more papers for better grounding.")
        if n_details.get("paper_coverage", 0) < 0.1:
            recs.append("Low paper coverage — narrative cites very few of the retrieved papers.")

        if not recs:
            recs.append("All metrics are within acceptable ranges. The pipeline is performing well.")

        return recs

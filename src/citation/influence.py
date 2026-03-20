"""Temporal influence scoring for identifying paradigm-shifting papers.

Combines multiple signals to score each paper's influence on the field:
- PageRank on the citation graph (structural importance)
- Citation burst detection (sudden increase in citations)
- Bridge score (connects otherwise disconnected clusters)
- Temporal precedence (early papers in a cluster that spawned many followers)
"""

from __future__ import annotations
import logging
from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np

from src.models.paper import Paper
from src.citation.graph import CitationGraph

logger = logging.getLogger(__name__)


class InfluenceScorer:
    """Computes influence scores for papers in a citation graph."""

    def __init__(self, citation_graph: CitationGraph):
        self.cg = citation_graph
        self.scores: dict[str, dict[str, float]] = {}

    def compute_all(self, papers: list[Paper]) -> dict[str, dict[str, float]]:
        """Compute all influence metrics for each paper.

        Returns dict mapping paper_id -> {metric_name: score}.
        """
        pagerank = self._pagerank()
        hub_scores, authority_scores = self._hits()
        bridge = self._bridge_scores()
        temporal = self._temporal_pioneer_scores(papers)
        burst = self._citation_burst_scores(papers)

        self.scores = {}
        for p in papers:
            pid = p.paper_id
            pr = pagerank.get(pid, 0.0)
            auth = authority_scores.get(pid, 0.0)
            br = bridge.get(pid, 0.0)
            tp = temporal.get(pid, 0.0)
            bu = burst.get(pid, 0.0)

            # Composite influence score (weighted combination)
            composite = (
                0.30 * pr +
                0.25 * auth +
                0.20 * br +
                0.15 * tp +
                0.10 * bu
            )

            self.scores[pid] = {
                "pagerank": pr,
                "authority": auth,
                "bridge": br,
                "temporal_pioneer": tp,
                "citation_burst": bu,
                "composite": composite,
            }

        logger.info(f"Computed influence scores for {len(self.scores)} papers")
        return self.scores

    def get_top_influential(self, n: int = 10) -> list[tuple[str, float]]:
        """Return top-N papers by composite influence score."""
        ranked = sorted(
            self.scores.items(),
            key=lambda x: x[1]["composite"],
            reverse=True,
        )
        return [(pid, s["composite"]) for pid, s in ranked[:n]]

    def get_paradigm_shifters(self, papers: list[Paper], threshold: float = 0.7) -> list[Paper]:
        """Identify papers that likely caused paradigm shifts.

        A paradigm shifter scores high on bridge + temporal_pioneer
        (introduces ideas that cross cluster boundaries early).
        """
        if not self.scores:
            self.compute_all(papers)

        shifters = []
        for p in papers:
            s = self.scores.get(p.paper_id, {})
            shift_score = 0.5 * s.get("bridge", 0) + 0.5 * s.get("temporal_pioneer", 0)
            if shift_score >= threshold:
                shifters.append(p)

        shifters.sort(key=lambda p: self.scores[p.paper_id]["composite"], reverse=True)
        return shifters

    def _pagerank(self) -> dict[str, float]:
        """PageRank on the citation graph."""
        g = self.cg.graph
        if g.number_of_nodes() == 0:
            return {}
        try:
            pr = nx.pagerank(g, alpha=0.85, max_iter=100)
            # Normalize to [0, 1]
            max_pr = max(pr.values()) if pr else 1.0
            return {k: v / max_pr if max_pr > 0 else 0 for k, v in pr.items()}
        except Exception as e:
            logger.warning(f"PageRank failed: {e}")
            return {}

    def _hits(self) -> tuple[dict[str, float], dict[str, float]]:
        """HITS algorithm — hub and authority scores."""
        g = self.cg.graph
        if g.number_of_nodes() == 0:
            return {}, {}
        try:
            hubs, authorities = nx.hits(g, max_iter=100, normalized=True)
            max_h = max(hubs.values()) if hubs else 1.0
            max_a = max(authorities.values()) if authorities else 1.0
            hubs = {k: v / max_h if max_h > 0 else 0 for k, v in hubs.items()}
            authorities = {k: v / max_a if max_a > 0 else 0 for k, v in authorities.items()}
            return hubs, authorities
        except Exception as e:
            logger.warning(f"HITS failed: {e}")
            return {}, {}

    def _bridge_scores(self) -> dict[str, float]:
        """Score papers that bridge different clusters in the citation graph.

        A paper is a bridge if it cites or is cited by papers in multiple clusters.
        """
        scores = {}
        for node in self.cg.graph.nodes:
            neighbor_clusters = set()
            for pred in self.cg.graph.predecessors(node):
                cl = self.cg.graph.nodes[pred].get("cluster_id", -1)
                if cl != -1:
                    neighbor_clusters.add(cl)
            for succ in self.cg.graph.successors(node):
                cl = self.cg.graph.nodes[succ].get("cluster_id", -1)
                if cl != -1:
                    neighbor_clusters.add(cl)

            own_cluster = self.cg.graph.nodes[node].get("cluster_id", -1)
            if own_cluster != -1:
                neighbor_clusters.discard(own_cluster)

            scores[node] = len(neighbor_clusters)

        max_score = max(scores.values()) if scores else 1.0
        return {k: v / max_score if max_score > 0 else 0 for k, v in scores.items()}

    def _temporal_pioneer_scores(self, papers: list[Paper]) -> dict[str, float]:
        """Score papers that are early entrants in their cluster.

        Papers published earliest in their cluster get higher scores,
        weighted by their in-degree (how many followers they attracted).
        """
        cluster_papers: dict[int, list[Paper]] = defaultdict(list)
        for p in papers:
            if p.cluster_id != -1:
                cluster_papers[p.cluster_id].append(p)

        scores = {}
        for cid, cpapers in cluster_papers.items():
            years = [p.year for p in cpapers if p.year]
            if not years:
                continue
            min_year = min(years)
            max_year = max(years)
            year_span = max(max_year - min_year, 1)

            for p in cpapers:
                if p.year is None:
                    scores[p.paper_id] = 0.0
                    continue
                # How early in the cluster (1.0 = earliest, 0.0 = latest)
                earliness = 1.0 - (p.year - min_year) / year_span
                # Weight by internal citations
                in_deg = self.cg.in_degree(p.paper_id)
                scores[p.paper_id] = earliness * (1 + np.log1p(in_deg))

        max_score = max(scores.values()) if scores else 1.0
        return {k: v / max_score if max_score > 0 else 0 for k, v in scores.items()}

    def _citation_burst_scores(self, papers: list[Paper]) -> dict[str, float]:
        """Score papers with high citations relative to their age.

        Newer papers with high citation counts indicate a burst.
        """
        current_year = 2026
        scores = {}
        for p in papers:
            if not p.year or p.citation_count == 0:
                scores[p.paper_id] = 0.0
                continue
            age = max(current_year - p.year, 1)
            scores[p.paper_id] = p.citation_count / age

        max_score = max(scores.values()) if scores else 1.0
        return {k: v / max_score if max_score > 0 else 0 for k, v in scores.items()}

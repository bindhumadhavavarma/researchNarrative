"""Citation graph construction and analysis.

Builds a directed citation graph from paper reference/citation data
and provides graph-level analytics for the ResearchNarrative pipeline.
"""

from __future__ import annotations
import logging
from typing import Optional
from collections import defaultdict

import networkx as nx
import numpy as np

from src.models.paper import Paper

logger = logging.getLogger(__name__)


class CitationGraph:
    """Directed citation graph where edges represent citation relationships.

    Edge direction: citing_paper -> cited_paper (A -> B means A cites B).
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._paper_map: dict[str, Paper] = {}

    def build(self, papers: list[Paper]) -> nx.DiGraph:
        """Construct the citation graph from a paper collection.

        Only includes edges between papers that are in the collection
        (internal citation network).
        """
        self._paper_map = {}

        # Build lookup tables for all IDs a paper might be known by
        id_to_paper_id: dict[str, str] = {}
        for p in papers:
            self._paper_map[p.paper_id] = p
            id_to_paper_id[p.paper_id] = p.paper_id
            if p.s2_id:
                id_to_paper_id[p.s2_id] = p.paper_id
            if p.arxiv_id:
                id_to_paper_id[f"arxiv:{p.arxiv_id}"] = p.paper_id
                id_to_paper_id[p.arxiv_id] = p.paper_id

        self.graph = nx.DiGraph()

        for p in papers:
            self.graph.add_node(p.paper_id, **{
                "title": p.title,
                "year": p.year or 0,
                "citation_count": p.citation_count,
                "cluster_id": p.cluster_id,
                "cluster_label": p.cluster_label,
            })

        edge_count = 0
        for p in papers:
            for ref_id in p.references:
                target = id_to_paper_id.get(ref_id)
                if target and target != p.paper_id:
                    self.graph.add_edge(p.paper_id, target)
                    edge_count += 1

            for citer_id in p.cited_by:
                source = id_to_paper_id.get(citer_id)
                if source and source != p.paper_id:
                    self.graph.add_edge(source, p.paper_id)
                    edge_count += 1

        # Deduplicate edges (add_edge is idempotent for same src,dst)
        logger.info(
            f"Citation graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return self.graph

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        return self._paper_map.get(paper_id)

    @property
    def nodes(self) -> list[str]:
        return list(self.graph.nodes)

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self.graph.edges)

    def in_degree(self, paper_id: str) -> int:
        """Number of papers in the collection that cite this paper."""
        return self.graph.in_degree(paper_id) if paper_id in self.graph else 0

    def out_degree(self, paper_id: str) -> int:
        """Number of papers in the collection this paper cites."""
        return self.graph.out_degree(paper_id) if paper_id in self.graph else 0

    def predecessors(self, paper_id: str) -> list[str]:
        """Papers that cite this paper (incoming edges)."""
        if paper_id not in self.graph:
            return []
        return list(self.graph.predecessors(paper_id))

    def successors(self, paper_id: str) -> list[str]:
        """Papers this paper cites (outgoing edges)."""
        if paper_id not in self.graph:
            return []
        return list(self.graph.successors(paper_id))

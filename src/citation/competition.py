"""Competition detection between research threads.

Analyzes cross-cluster citation patterns to identify:
- Competing threads (mutual awareness but divergent approaches)
- Complementary threads (building on each other)
- Dominant threads at different time periods
"""

from __future__ import annotations
import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from src.models.paper import Paper
from src.citation.graph import CitationGraph

logger = logging.getLogger(__name__)


class CompetitionDetector:
    """Detects competitive and complementary relationships between clusters."""

    def __init__(self, citation_graph: CitationGraph):
        self.cg = citation_graph

    def analyze(
        self,
        papers: list[Paper],
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
    ) -> dict:
        """Run full competition analysis.

        Returns:
            Dictionary with cross_citation_matrix, competition_pairs,
            complementary_pairs, and dominance_timeline.
        """
        real_clusters = {cid: ps for cid, ps in clusters.items() if cid != -1}
        if len(real_clusters) < 2:
            logger.info("Fewer than 2 clusters — skipping competition analysis")
            return {
                "cross_citation_matrix": {},
                "competition_pairs": [],
                "complementary_pairs": [],
                "dominance_timeline": {},
                "cluster_labels": cluster_labels,
            }

        matrix = self._cross_citation_matrix(real_clusters)
        competition = self._find_competition(matrix, cluster_labels)
        complementary = self._find_complementary(matrix, cluster_labels)
        dominance = self._dominance_timeline(papers, real_clusters, cluster_labels)

        logger.info(
            f"Competition analysis: {len(competition)} competing pairs, "
            f"{len(complementary)} complementary pairs"
        )

        return {
            "cross_citation_matrix": matrix,
            "competition_pairs": competition,
            "complementary_pairs": complementary,
            "dominance_timeline": dominance,
            "cluster_labels": cluster_labels,
        }

    def _cross_citation_matrix(
        self, clusters: dict[int, list[Paper]]
    ) -> dict[tuple[int, int], int]:
        """Build a matrix of citation counts between every pair of clusters.

        matrix[(i, j)] = number of citations from papers in cluster i
                         to papers in cluster j.
        """
        paper_to_cluster = {}
        for cid, papers in clusters.items():
            for p in papers:
                paper_to_cluster[p.paper_id] = cid

        matrix: dict[tuple[int, int], int] = defaultdict(int)
        cluster_ids = sorted(clusters.keys())

        for cid_from in cluster_ids:
            for p in clusters[cid_from]:
                for ref_id in self.cg.successors(p.paper_id):
                    cid_to = paper_to_cluster.get(ref_id)
                    if cid_to is not None and cid_to != cid_from:
                        matrix[(cid_from, cid_to)] += 1

        return dict(matrix)

    def _find_competition(
        self,
        matrix: dict[tuple[int, int], int],
        labels: dict[int, str],
    ) -> list[dict]:
        """Identify competing cluster pairs.

        Competition = mutual citation (both cite each other) but with
        asymmetry indicating rivalry rather than pure building-upon.
        """
        pairs_seen = set()
        competition_pairs = []

        for (ci, cj), count_ij in matrix.items():
            pair = (min(ci, cj), max(ci, cj))
            if pair in pairs_seen:
                continue
            pairs_seen.add(pair)

            count_ji = matrix.get((cj, ci), 0)
            if count_ij == 0 and count_ji == 0:
                continue

            total = count_ij + count_ji
            # Asymmetry ratio: 0 = perfectly balanced, 1 = one-directional
            asymmetry = abs(count_ij - count_ji) / total if total > 0 else 0

            # Competition: both directions exist and reasonably balanced
            if count_ij > 0 and count_ji > 0 and asymmetry < 0.7:
                competition_pairs.append({
                    "cluster_a": ci,
                    "cluster_b": cj,
                    "label_a": labels.get(ci, f"Thread {ci}"),
                    "label_b": labels.get(cj, f"Thread {cj}"),
                    "a_cites_b": count_ij,
                    "b_cites_a": count_ji,
                    "total_cross_citations": total,
                    "asymmetry": round(asymmetry, 3),
                    "relationship": "competing",
                })

        competition_pairs.sort(key=lambda x: x["total_cross_citations"], reverse=True)
        return competition_pairs

    def _find_complementary(
        self,
        matrix: dict[tuple[int, int], int],
        labels: dict[int, str],
    ) -> list[dict]:
        """Identify complementary cluster pairs.

        Complementary = heavily one-directional citations (one builds on the other).
        """
        pairs_seen = set()
        complementary_pairs = []

        for (ci, cj), count_ij in matrix.items():
            pair = (min(ci, cj), max(ci, cj))
            if pair in pairs_seen:
                continue
            pairs_seen.add(pair)

            count_ji = matrix.get((cj, ci), 0)
            total = count_ij + count_ji
            if total < 2:
                continue

            asymmetry = abs(count_ij - count_ji) / total if total > 0 else 0

            # Complementary: highly asymmetric (one builds on the other)
            if asymmetry >= 0.7:
                builder = ci if count_ij > count_ji else cj
                foundation = cj if count_ij > count_ji else ci
                complementary_pairs.append({
                    "foundation_cluster": foundation,
                    "builder_cluster": builder,
                    "foundation_label": labels.get(foundation, f"Thread {foundation}"),
                    "builder_label": labels.get(builder, f"Thread {builder}"),
                    "foundation_to_builder": matrix.get((foundation, builder), 0),
                    "builder_to_foundation": matrix.get((builder, foundation), 0),
                    "total_cross_citations": total,
                    "asymmetry": round(asymmetry, 3),
                    "relationship": "complementary",
                })

        complementary_pairs.sort(key=lambda x: x["total_cross_citations"], reverse=True)
        return complementary_pairs

    def _dominance_timeline(
        self,
        papers: list[Paper],
        clusters: dict[int, list[Paper]],
        labels: dict[int, str],
    ) -> dict[int, list[dict]]:
        """Track which cluster dominates at each time period.

        Dominance is measured by share of papers and share of citations per year.
        """
        all_years = sorted(set(p.year for p in papers if p.year))
        if not all_years:
            return {}

        cluster_ids = sorted(clusters.keys())
        timeline: dict[int, list[dict]] = {}

        for year in all_years:
            year_data = []
            total_papers_year = 0
            total_cites_year = 0

            for cid in cluster_ids:
                year_papers = [p for p in clusters[cid] if p.year == year]
                n_papers = len(year_papers)
                n_cites = sum(p.citation_count for p in year_papers)
                total_papers_year += n_papers
                total_cites_year += n_cites
                year_data.append({
                    "cluster_id": cid,
                    "label": labels.get(cid, f"Thread {cid}"),
                    "papers": n_papers,
                    "citations": n_cites,
                })

            for entry in year_data:
                entry["paper_share"] = round(
                    entry["papers"] / total_papers_year, 3
                ) if total_papers_year > 0 else 0
                entry["citation_share"] = round(
                    entry["citations"] / total_cites_year, 3
                ) if total_cites_year > 0 else 0

            timeline[year] = year_data

        return timeline

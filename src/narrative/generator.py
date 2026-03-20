"""RAG-based narrative generation engine.

Generates structured research storylines grounded in retrieved paper data,
with explicit citations and multi-section organization.
"""

from __future__ import annotations
import logging
from typing import Optional, Any

from src.models.paper import Paper
from src.config import LLM_TEMPERATURE, HAS_LLM
from src.utils.llm import get_llm_client, get_model_name
from src.citation.graph import CitationGraph

logger = logging.getLogger(__name__)


NARRATIVE_SYSTEM_PROMPT = """\
You are an expert academic researcher and science writer. Your task is to \
generate a structured research narrative that tells the "story" of a research \
area based on the provided papers.

RULES:
1. Every factual claim MUST cite at least one paper using [Author, Year] format.
2. Organize the narrative chronologically within each section.
3. Highlight competing approaches and explain how ideas evolved.
4. Be concise but comprehensive — aim for depth, not breadth.
5. Use the EXACT paper information provided; do NOT hallucinate papers or facts.
6. Write in an academic but accessible style.
"""

NARRATIVE_TEMPLATE = """\
Generate a structured research narrative for the topic: "{topic}"

I have organized {total_papers} papers into {num_clusters} research threads.

{cluster_summaries}

{citation_analysis_block}

Write a research narrative with these sections:

## 1. Origins & Foundations
Describe how this research area began. Which were the seminal papers? \
What problems motivated the initial work? Use the influence scores to \
identify the most foundational papers.

## 2. Major Research Threads
For each identified thread, explain what it investigates, key contributions, \
and how it relates to other threads.

## 3. Competing Approaches
Use the competition analysis data provided to identify pairs or groups of \
approaches that compete or offer alternatives. Explain the trade-offs and \
cross-citation patterns.

## 4. Evolution & Paradigm Shifts
Trace how dominant approaches changed over time. Highlight paradigm-shifting \
papers (high bridge + pioneer scores). Use the dominance timeline to show \
which threads gained or lost influence.

## 5. Current State & Open Problems
What is the frontier right now? What problems remain unsolved?

IMPORTANT: Cite papers as [FirstAuthor et al., Year] throughout. \
Use ONLY the papers provided below.
"""


def _build_cluster_summary(
    cluster_id: int,
    label: str,
    papers: list[Paper],
) -> str:
    """Build a summary block for one cluster to include in the prompt."""
    sorted_papers = sorted(papers, key=lambda p: (p.year or 0, -p.citation_count))
    lines = [f"### Thread {cluster_id}: {label} ({len(papers)} papers)"]

    for p in sorted_papers[:15]:
        first_author = p.authors[0].name if p.authors else "Unknown"
        if len(p.authors) > 1:
            first_author += " et al."
        abstract_snippet = (p.abstract[:300] + "...") if len(p.abstract) > 300 else p.abstract
        lines.append(
            f"- [{first_author}, {p.year}] \"{p.title}\" "
            f"(cited {p.citation_count}x): {abstract_snippet}"
        )

    return "\n".join(lines)


class NarrativeGenerator:
    """Generates research narratives using RAG with LLM."""

    def generate(
        self,
        topic: str,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
        influence_scores: Optional[dict] = None,
        competition_analysis: Optional[dict] = None,
        citation_graph: Optional[CitationGraph] = None,
    ) -> str:
        """Generate a full research narrative.

        Args:
            topic: The research topic.
            clusters: Mapping of cluster_id -> papers.
            cluster_labels: Mapping of cluster_id -> human-readable label.
            influence_scores: Per-paper influence metrics from InfluenceScorer.
            competition_analysis: Competition/complementary data from CompetitionDetector.
            citation_graph: The CitationGraph object.

        Returns:
            Generated narrative as a markdown string.
        """
        if not HAS_LLM:
            return self._generate_without_llm(
                topic, clusters, cluster_labels, influence_scores, competition_analysis
            )

        return self._generate_with_llm(
            topic, clusters, cluster_labels, influence_scores, competition_analysis, citation_graph
        )

    def _generate_with_llm(
        self,
        topic: str,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
        influence_scores: Optional[dict] = None,
        competition_analysis: Optional[dict] = None,
        citation_graph: Optional[CitationGraph] = None,
    ) -> str:
        """Generate narrative using OpenAI API."""
        try:
            client = get_llm_client()
            model = get_model_name()

            real_clusters = {
                cid: papers for cid, papers in clusters.items() if cid != -1
            }

            cluster_summaries = "\n\n".join(
                _build_cluster_summary(cid, cluster_labels.get(cid, f"Thread {cid}"), papers)
                for cid, papers in sorted(real_clusters.items())
            )

            citation_block = self._build_citation_analysis_block(
                real_clusters, influence_scores, competition_analysis, citation_graph
            )

            total_papers = sum(len(p) for p in real_clusters.values())
            user_prompt = NARRATIVE_TEMPLATE.format(
                topic=topic,
                total_papers=total_papers,
                num_clusters=len(real_clusters),
                cluster_summaries=cluster_summaries,
                citation_analysis_block=citation_block,
            )

            response = client.chat.completions.create(
                model=model,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4096,
            )

            narrative = response.choices[0].message.content
            logger.info(f"Generated narrative: {len(narrative)} characters")
            return narrative

        except Exception as e:
            logger.error(f"LLM narrative generation failed: {e}")
            return self._generate_without_llm(
                topic, clusters, cluster_labels, influence_scores, competition_analysis
            )

    def _build_citation_analysis_block(
        self,
        clusters: dict[int, list[Paper]],
        influence_scores: Optional[dict],
        competition_analysis: Optional[dict],
        citation_graph: Optional[CitationGraph],
    ) -> str:
        """Build a citation analysis summary block for the LLM prompt."""
        parts = []

        if influence_scores:
            ranked = sorted(
                influence_scores.items(),
                key=lambda x: x[1].get("composite", 0),
                reverse=True,
            )[:10]
            lines = ["### Most Influential Papers (by composite influence score)"]
            for pid, scores in ranked:
                paper = citation_graph.get_paper(pid) if citation_graph else None
                if paper:
                    first_author = paper.authors[0].name if paper.authors else "Unknown"
                    lines.append(
                        f"- [{first_author}, {paper.year}] \"{paper.title}\" — "
                        f"PageRank: {scores['pagerank']:.3f}, "
                        f"Authority: {scores['authority']:.3f}, "
                        f"Bridge: {scores['bridge']:.3f}, "
                        f"Pioneer: {scores['temporal_pioneer']:.3f}, "
                        f"Burst: {scores['citation_burst']:.3f}"
                    )
            parts.append("\n".join(lines))

        if competition_analysis:
            comp_pairs = competition_analysis.get("competition_pairs", [])
            if comp_pairs:
                lines = ["### Competing Thread Pairs"]
                for cp in comp_pairs[:5]:
                    lines.append(
                        f"- \"{cp['label_a']}\" vs \"{cp['label_b']}\": "
                        f"{cp['a_cites_b']} cross-citations A→B, "
                        f"{cp['b_cites_a']} B→A (asymmetry: {cp['asymmetry']})"
                    )
                parts.append("\n".join(lines))

            comp_pairs = competition_analysis.get("complementary_pairs", [])
            if comp_pairs:
                lines = ["### Complementary Thread Pairs"]
                for cp in comp_pairs[:5]:
                    lines.append(
                        f"- \"{cp['foundation_label']}\" (foundation) → "
                        f"\"{cp['builder_label']}\" (builds upon): "
                        f"{cp['builder_to_foundation']} citations toward foundation"
                    )
                parts.append("\n".join(lines))

        return "\n\n".join(parts) if parts else "(No citation graph data available)"

    def _generate_without_llm(
        self,
        topic: str,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
        influence_scores: Optional[dict] = None,
        competition_analysis: Optional[dict] = None,
    ) -> str:
        """Generate a structured summary without LLM (template-based fallback)."""
        lines = [
            f"# Research Narrative: {topic}",
            "",
            f"*Analysis of {sum(len(p) for p in clusters.values())} papers "
            f"across {len([c for c in clusters if c != -1])} research threads.*",
            "",
        ]

        all_papers = [p for papers in clusters.values() for p in papers]
        all_papers.sort(key=lambda p: (p.year or 9999))

        lines.append("## 1. Origins & Foundations")
        lines.append("")
        earliest = all_papers[:5]
        for p in earliest:
            first_author = p.authors[0].name if p.authors else "Unknown"
            lines.append(
                f"- **[{first_author}, {p.year}]** \"{p.title}\" — "
                f"Cited {p.citation_count} times."
            )
        lines.append("")

        lines.append("## 2. Major Research Threads")
        lines.append("")
        for cid, papers in sorted(clusters.items()):
            if cid == -1:
                continue
            label = cluster_labels.get(cid, f"Thread {cid}")
            papers_sorted = sorted(papers, key=lambda p: -p.citation_count)
            years = [p.year for p in papers if p.year]
            year_range = f"{min(years)}-{max(years)}" if years else "N/A"

            lines.append(f"### Thread {cid}: {label}")
            lines.append(f"*{len(papers)} papers, {year_range}*")
            lines.append("")
            for p in papers_sorted[:5]:
                first_author = p.authors[0].name if p.authors else "Unknown"
                lines.append(
                    f"- [{first_author} et al., {p.year}] \"{p.title}\" "
                    f"(citations: {p.citation_count})"
                )
            lines.append("")

        # Influential papers (use influence scores if available)
        lines.append("## 3. Most Influential Papers")
        lines.append("")
        if influence_scores:
            ranked = sorted(
                [(p, influence_scores.get(p.paper_id, {}).get("composite", 0)) for p in all_papers],
                key=lambda x: -x[1],
            )[:10]
            for i, (p, score) in enumerate(ranked, 1):
                first_author = p.authors[0].name if p.authors else "Unknown"
                lines.append(
                    f"{i}. **[{first_author} et al., {p.year}]** \"{p.title}\" "
                    f"— Influence: {score:.3f}, Citations: {p.citation_count} "
                    f"(Thread: {p.cluster_label})"
                )
        else:
            top_cited = sorted(all_papers, key=lambda p: -p.citation_count)[:10]
            for i, p in enumerate(top_cited, 1):
                first_author = p.authors[0].name if p.authors else "Unknown"
                lines.append(
                    f"{i}. **[{first_author} et al., {p.year}]** \"{p.title}\" "
                    f"— {p.citation_count} citations (Thread: {p.cluster_label})"
                )
        lines.append("")

        # Competition section
        if competition_analysis:
            comp_pairs = competition_analysis.get("competition_pairs", [])
            complementary = competition_analysis.get("complementary_pairs", [])

            if comp_pairs:
                lines.append("## 4. Competing Approaches")
                lines.append("")
                for cp in comp_pairs[:5]:
                    lines.append(
                        f"- **\"{cp['label_a']}\"** vs **\"{cp['label_b']}\"** — "
                        f"{cp['total_cross_citations']} cross-citations "
                        f"(asymmetry: {cp['asymmetry']})"
                    )
                lines.append("")

            if complementary:
                lines.append("## 5. Complementary Threads")
                lines.append("")
                for cp in complementary[:5]:
                    lines.append(
                        f"- **\"{cp['foundation_label']}\"** → **\"{cp['builder_label']}\"** — "
                        f"The latter builds on the former ({cp['builder_to_foundation']} citations)"
                    )
                lines.append("")

        # Timeline
        lines.append("## 6. Timeline")
        lines.append("")
        all_years = [p.year for p in all_papers if p.year]
        if all_years:
            from collections import Counter
            year_counts = Counter(all_years)
            for year in sorted(year_counts.keys()):
                lines.append(f"- **{year}**: {year_counts[year]} papers")
        lines.append("")

        lines.append("---")
        lines.append("*Note: Full narrative generation requires an OpenAI API key. "
                     "Set OPENAI_API_KEY in your .env file for richer narratives.*")

        return "\n".join(lines)

    def generate_thread_narrative(
        self,
        thread_label: str,
        papers: list[Paper],
    ) -> str:
        """Generate a focused narrative for a single research thread."""
        if not HAS_LLM:
            return self._thread_summary_fallback(thread_label, papers)

        try:
            client = get_llm_client()
            model = get_model_name()

            sorted_papers = sorted(papers, key=lambda p: (p.year or 0))
            paper_block = "\n".join(
                f"- [{p.authors[0].name if p.authors else 'Unknown'} et al., {p.year}] "
                f"\"{p.title}\": {p.abstract[:200]}..."
                for p in sorted_papers[:20]
            )

            response = client.chat.completions.create(
                model=model,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Write a 2-3 paragraph narrative about the research thread "
                            f"\"{thread_label}\" based on these papers:\n\n{paper_block}\n\n"
                            f"Focus on: how it started, key advances, and current state. "
                            f"Cite papers as [Author, Year]."
                        ),
                    },
                ],
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Thread narrative failed: {e}")
            return self._thread_summary_fallback(thread_label, papers)

    def _thread_summary_fallback(self, label: str, papers: list[Paper]) -> str:
        sorted_papers = sorted(papers, key=lambda p: (p.year or 0))
        lines = [f"**{label}** — {len(papers)} papers\n"]
        for p in sorted_papers[:10]:
            first_author = p.authors[0].name if p.authors else "Unknown"
            lines.append(
                f"- [{first_author}, {p.year}] \"{p.title}\" ({p.citation_count} citations)"
            )
        return "\n".join(lines)

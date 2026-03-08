"""RAG-based narrative generation engine.

Generates structured research storylines grounded in retrieved paper data,
with explicit citations and multi-section organization.
"""

from __future__ import annotations
import logging
from typing import Optional

from src.models.paper import Paper
from src.config import LLM_TEMPERATURE, HAS_LLM
from src.utils.llm import get_llm_client, get_model_name

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

Write a research narrative with these sections:

## 1. Origins & Foundations
Describe how this research area began. Which were the seminal papers? \
What problems motivated the initial work?

## 2. Major Research Threads
For each identified thread, explain what it investigates, key contributions, \
and how it relates to other threads.

## 3. Competing Approaches
Identify pairs or groups of approaches that compete or offer alternatives. \
Explain the trade-offs.

## 4. Evolution & Paradigm Shifts
Trace how dominant approaches changed over time. What caused shifts?

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
    ) -> str:
        """Generate a full research narrative.

        Args:
            topic: The research topic.
            clusters: Mapping of cluster_id -> papers.
            cluster_labels: Mapping of cluster_id -> human-readable label.

        Returns:
            Generated narrative as a markdown string.
        """
        if not HAS_LLM:
            return self._generate_without_llm(topic, clusters, cluster_labels)

        return self._generate_with_llm(topic, clusters, cluster_labels)

    def _generate_with_llm(
        self,
        topic: str,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
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

            total_papers = sum(len(p) for p in real_clusters.values())
            user_prompt = NARRATIVE_TEMPLATE.format(
                topic=topic,
                total_papers=total_papers,
                num_clusters=len(real_clusters),
                cluster_summaries=cluster_summaries,
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
            return self._generate_without_llm(topic, clusters, cluster_labels)

    def _generate_without_llm(
        self,
        topic: str,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
    ) -> str:
        """Generate a structured summary without LLM (template-based fallback)."""
        lines = [
            f"# Research Narrative: {topic}",
            "",
            f"*Analysis of {sum(len(p) for p in clusters.values())} papers "
            f"across {len([c for c in clusters if c != -1])} research threads.*",
            "",
        ]

        # Origins
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

        # Threads
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

        # Most cited
        lines.append("## 3. Most Influential Papers")
        lines.append("")
        top_cited = sorted(all_papers, key=lambda p: -p.citation_count)[:10]
        for i, p in enumerate(top_cited, 1):
            first_author = p.authors[0].name if p.authors else "Unknown"
            lines.append(
                f"{i}. **[{first_author} et al., {p.year}]** \"{p.title}\" "
                f"— {p.citation_count} citations (Thread: {p.cluster_label})"
            )
        lines.append("")

        # Timeline
        lines.append("## 4. Timeline")
        lines.append("")
        if years:
            from collections import Counter
            year_counts = Counter(p.year for p in all_papers if p.year)
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

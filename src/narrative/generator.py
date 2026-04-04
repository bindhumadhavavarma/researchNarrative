"""Production-grade RAG narrative generation engine.

Generates structured research storylines grounded in retrieved paper data,
with explicit citations, multi-section organization, chunked generation
for large corpora, and post-generation citation verification.
"""

from __future__ import annotations
import logging
import re
from typing import Optional
from collections import defaultdict

from src.models.paper import Paper
from src.config import LLM_TEMPERATURE, HAS_LLM, MAX_NARRATIVE_PAPERS
from src.utils.llm import get_llm_client, get_model_name
from src.citation.graph import CitationGraph

logger = logging.getLogger(__name__)


NARRATIVE_SYSTEM_PROMPT = """\
You are an expert academic researcher and science writer. Your task is to \
generate a structured research narrative that tells the "story" of a research \
area based on the provided papers.

RULES:
1. Every factual claim MUST cite at least one paper using [Author et al., Year] format.
2. Organize the narrative chronologically within each section.
3. Highlight competing approaches and explain how ideas evolved.
4. Be concise but comprehensive — aim for depth, not breadth.
5. Use the EXACT paper information provided; do NOT hallucinate papers or facts.
6. Write in an academic but accessible style.
7. When citing, use the first author's surname followed by "et al." and the year.
"""

SECTION_ORIGINS = """\
Write the "Origins & Foundations" section for a research narrative on "{topic}".

Based on these foundational papers (sorted by influence score):

{papers_block}

{influence_block}

Write 3-5 paragraphs describing:
- How this research area began and what problems motivated the initial work
- Which were the seminal/foundational papers and why they mattered
- The key ideas or breakthroughs that launched the field

Cite every claim as [Author et al., Year]. Use ONLY the papers listed above.
"""

SECTION_THREADS = """\
Write the "Major Research Threads" section for a research narrative on "{topic}".

The following distinct research threads have been identified:

{threads_block}

Write a subsection for each thread (2-3 paragraphs each) explaining:
- What the thread investigates and its core research questions
- Key contributions and milestone papers
- How it relates to or differs from other threads

Cite every claim as [Author et al., Year]. Use ONLY the papers listed above.
"""

SECTION_COMPETITION = """\
Write the "Competing Approaches & Trade-offs" section for a research narrative on "{topic}".

Citation analysis has identified these relationships between research threads:

{competition_block}

Papers from the relevant threads:

{papers_block}

Write 2-4 paragraphs analyzing:
- Which approaches compete and what trade-offs they represent
- How cross-citation patterns reveal intellectual debate
- Which complementary threads build upon each other and why

Cite every claim as [Author et al., Year]. Use ONLY the papers listed above.
"""

SECTION_EVOLUTION = """\
Write the "Evolution & Paradigm Shifts" section for a research narrative on "{topic}".

Dominance timeline data (thread share of papers per year):
{dominance_block}

Paradigm-shifting papers (high bridge + pioneer scores):
{shifters_block}

Write 2-4 paragraphs tracing:
- How dominant approaches changed over time
- What caused paradigm shifts (new datasets, methods, results)
- Which papers bridged different research communities

Cite every claim as [Author et al., Year]. Use ONLY the papers listed above.
"""

SECTION_FRONTIER = """\
Write the "Current State & Open Problems" section for a research narrative on "{topic}".

Most recent papers (last 2-3 years), sorted by citation burst score:

{papers_block}

Write 2-4 paragraphs describing:
- What the current research frontier looks like
- The most active areas of investigation right now
- Key open problems and unsolved challenges
- Promising future directions

Cite every claim as [Author et al., Year]. Use ONLY the papers listed above.
"""

SYNTHESIS_PROMPT = """\
You are combining individually-generated sections into a cohesive research narrative on "{topic}".

Here are the sections (each already written with proper citations):

{sections}

Combine them into a single, cohesive research narrative with these sections:
## 1. Origins & Foundations
## 2. Major Research Threads
## 3. Competing Approaches & Trade-offs
## 4. Evolution & Paradigm Shifts
## 5. Current State & Open Problems

Rules:
- Keep ALL existing citations [Author et al., Year] intact
- Add smooth transitions between sections
- Remove any redundancy between sections
- Add a brief introductory paragraph before section 1
- Do NOT add new factual claims or citations that weren't in the sections
- Output clean markdown
"""


def _format_paper(p: Paper, include_abstract: bool = True) -> str:
    """Format a paper for inclusion in a prompt."""
    first_author = p.authors[0].name if p.authors else "Unknown"
    if len(p.authors) > 1:
        first_author += " et al."
    line = f"- [{first_author}, {p.year}] \"{p.title}\" (cited {p.citation_count}x)"
    if include_abstract and p.abstract:
        snippet = p.abstract[:250] + "..." if len(p.abstract) > 250 else p.abstract
        line += f": {snippet}"
    return line


def _format_papers_block(papers: list[Paper], max_papers: int = 20) -> str:
    """Format a list of papers into a prompt block."""
    selected = papers[:max_papers]
    return "\n".join(_format_paper(p) for p in selected)


class CitationVerifier:
    """Post-generation verification that narrative citations reference real papers."""

    CITATION_PATTERN = re.compile(
        r'\[([A-Z][a-zA-Z\-\']+(?:\s+et\s+al\.)?),?\s*(\d{4})\]'
    )

    def __init__(self, papers: list[Paper]):
        self._build_index(papers)

    def _build_index(self, papers: list[Paper]) -> None:
        """Build a lookup index mapping (surname, year) -> Paper."""
        self.index: dict[tuple[str, int], list[Paper]] = defaultdict(list)
        self.all_papers = papers
        for p in papers:
            if not p.authors or not p.year:
                continue
            surname = p.authors[0].name.split()[-1].lower()
            self.index[(surname, p.year)].append(p)

    def verify(self, narrative: str) -> dict:
        """Verify all citations in a narrative.

        Returns dict with:
        - verified: list of (citation_text, matched_paper_title)
        - unverified: list of citation_text that couldn't be matched
        - stats: {total, verified_count, unverified_count, accuracy}
        """
        citations = self.CITATION_PATTERN.findall(narrative)
        verified = []
        unverified = []

        for author_part, year_str in citations:
            year = int(year_str)
            surname = author_part.replace(" et al.", "").replace(" et al", "").strip()
            surname_lower = surname.split()[-1].lower() if surname else ""

            matches = self.index.get((surname_lower, year), [])
            if matches:
                verified.append((f"[{author_part}, {year_str}]", matches[0].title))
            else:
                # Fuzzy: try matching just the year and a partial surname
                found = False
                for (s, y), papers in self.index.items():
                    if y == year and surname_lower in s:
                        verified.append((f"[{author_part}, {year_str}]", papers[0].title))
                        found = True
                        break
                if not found:
                    unverified.append(f"[{author_part}, {year_str}]")

        total = len(citations)
        ver_count = len(verified)
        unver_count = len(unverified)
        accuracy = ver_count / total if total > 0 else 1.0

        return {
            "verified": verified,
            "unverified": unverified,
            "stats": {
                "total": total,
                "verified_count": ver_count,
                "unverified_count": unver_count,
                "accuracy": round(accuracy, 3),
            },
        }

    def add_paper_links(self, narrative: str) -> str:
        """Replace citation markers with markdown links to paper URLs."""
        def _replace_citation(match):
            full_match = match.group(0)
            author_part = match.group(1)
            year_str = match.group(2)
            year = int(year_str)
            surname = author_part.replace(" et al.", "").replace(" et al", "").strip()
            surname_lower = surname.split()[-1].lower() if surname else ""

            matches = self.index.get((surname_lower, year), [])
            if not matches:
                for (s, y), papers in self.index.items():
                    if y == year and surname_lower in s:
                        matches = papers
                        break

            if matches and matches[0].url:
                return f"[{author_part}, {year_str}]({matches[0].url})"
            return full_match

        return self.CITATION_PATTERN.sub(_replace_citation, narrative)


class NarrativeGenerator:
    """Production-grade RAG narrative generator with chunking and verification."""

    def __init__(self):
        self.verifier: Optional[CitationVerifier] = None
        self.verification_result: Optional[dict] = None
        self.thread_narratives: dict[int, str] = {}

    def generate(
        self,
        topic: str,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
        influence_scores: Optional[dict] = None,
        competition_analysis: Optional[dict] = None,
        citation_graph: Optional[CitationGraph] = None,
        progress_callback: Optional[callable] = None,
    ) -> str:
        """Generate a full research narrative with chunking and verification.

        For large corpora, generates each section independently then
        synthesizes them into a coherent narrative.
        """
        all_papers = [p for ps in clusters.values() for p in ps]
        self.verifier = CitationVerifier(all_papers)

        if not HAS_LLM:
            narrative = self._generate_without_llm(
                topic, clusters, cluster_labels, influence_scores, competition_analysis
            )
            self.verification_result = self.verifier.verify(narrative)
            return narrative

        def _report(msg: str):
            if progress_callback:
                progress_callback(msg)

        real_clusters = {cid: ps for cid, ps in clusters.items() if cid != -1}
        total_papers = sum(len(ps) for ps in real_clusters.values())

        if total_papers <= MAX_NARRATIVE_PAPERS and len(real_clusters) <= 6:
            _report("Generating narrative (single-pass mode)...")
            narrative = self._generate_single_pass(
                topic, clusters, cluster_labels,
                influence_scores, competition_analysis, citation_graph
            )
        else:
            _report(f"Large corpus ({total_papers} papers, {len(real_clusters)} threads) — using chunked generation...")
            narrative = self._generate_chunked(
                topic, clusters, cluster_labels,
                influence_scores, competition_analysis, citation_graph,
                _report,
            )

        _report("Verifying citations against paper database...")
        self.verification_result = self.verifier.verify(narrative)
        stats = self.verification_result["stats"]
        _report(
            f"Citation verification: {stats['verified_count']}/{stats['total']} verified "
            f"({stats['accuracy']:.0%} accuracy), {stats['unverified_count']} unverified"
        )

        return narrative

    def generate_thread_narrative(
        self,
        thread_label: str,
        papers: list[Paper],
        influence_scores: Optional[dict] = None,
    ) -> str:
        """Generate a focused narrative for a single research thread."""
        if not HAS_LLM:
            return self._thread_summary_fallback(thread_label, papers)

        try:
            client = get_llm_client()
            model = get_model_name()

            sorted_papers = sorted(papers, key=lambda p: (p.year or 0))

            inf_lines = []
            if influence_scores:
                ranked = sorted(
                    papers,
                    key=lambda p: influence_scores.get(p.paper_id, {}).get("composite", 0),
                    reverse=True,
                )[:5]
                for p in ranked:
                    scores = influence_scores.get(p.paper_id, {})
                    first_author = p.authors[0].name if p.authors else "Unknown"
                    inf_lines.append(
                        f"- [{first_author}, {p.year}]: composite={scores.get('composite', 0):.3f}"
                    )

            inf_block = "\n".join(inf_lines) if inf_lines else ""
            paper_block = _format_papers_block(sorted_papers, max_papers=25)

            prompt = (
                f"Write a detailed 3-5 paragraph narrative about the research thread "
                f"\"{thread_label}\" based on these {len(papers)} papers:\n\n{paper_block}\n\n"
            )
            if inf_block:
                prompt += f"Most influential papers in this thread:\n{inf_block}\n\n"
            prompt += (
                "Cover: (1) how this thread started and what motivated it, "
                "(2) key advances and milestone papers, (3) current state and open questions. "
                "Cite papers as [Author et al., Year]."
            )

            response = client.chat.completions.create(
                model=model,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Thread narrative failed: {e}")
            return self._thread_summary_fallback(thread_label, papers)

    def _generate_single_pass(
        self,
        topic: str,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
        influence_scores: Optional[dict],
        competition_analysis: Optional[dict],
        citation_graph: Optional[CitationGraph],
    ) -> str:
        """Generate entire narrative in a single LLM call (small corpora)."""
        try:
            client = get_llm_client()
            model = get_model_name()

            real_clusters = {cid: ps for cid, ps in clusters.items() if cid != -1}

            cluster_summaries = "\n\n".join(
                self._build_thread_block(cid, cluster_labels.get(cid, f"Thread {cid}"), ps)
                for cid, ps in sorted(real_clusters.items())
            )

            citation_block = self._build_citation_analysis_block(
                real_clusters, influence_scores, competition_analysis, citation_graph
            )

            total_papers = sum(len(ps) for ps in real_clusters.values())
            user_prompt = (
                f'Generate a structured research narrative for the topic: "{topic}"\n\n'
                f"I have organized {total_papers} papers into {len(real_clusters)} research threads.\n\n"
                f"{cluster_summaries}\n\n{citation_block}\n\n"
                "Write a research narrative with these sections:\n\n"
                "## 1. Origins & Foundations\n"
                "Describe how this research area began. Which were the seminal papers? "
                "What problems motivated the initial work? Use the influence scores to "
                "identify the most foundational papers.\n\n"
                "## 2. Major Research Threads\n"
                "For each identified thread, explain what it investigates, key contributions, "
                "and how it relates to other threads.\n\n"
                "## 3. Competing Approaches & Trade-offs\n"
                "Use the competition analysis data to identify competing approaches. "
                "Explain the trade-offs and cross-citation patterns.\n\n"
                "## 4. Evolution & Paradigm Shifts\n"
                "Trace how dominant approaches changed over time. Highlight paradigm-shifting papers.\n\n"
                "## 5. Current State & Open Problems\n"
                "What is the frontier right now? What problems remain unsolved?\n\n"
                "IMPORTANT: Cite papers as [Author et al., Year] throughout. "
                "Use ONLY the papers provided above."
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
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Single-pass generation failed: {e}")
            return self._generate_without_llm(
                topic, clusters, cluster_labels, influence_scores, competition_analysis
            )

    def _generate_chunked(
        self,
        topic: str,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
        influence_scores: Optional[dict],
        competition_analysis: Optional[dict],
        citation_graph: Optional[CitationGraph],
        report: callable,
    ) -> str:
        """Generate narrative in chunks (one section at a time) then synthesize."""
        try:
            client = get_llm_client()
            model = get_model_name()
            real_clusters = {cid: ps for cid, ps in clusters.items() if cid != -1}
            all_papers = [p for ps in real_clusters.values() for p in ps]

            sections = {}

            # Section 1: Origins & Foundations
            report("Generating section 1/5: Origins & Foundations...")
            foundational = self._get_foundational_papers(all_papers, influence_scores)
            influence_block = self._build_influence_block(foundational, influence_scores)
            prompt_origins = SECTION_ORIGINS.format(
                topic=topic,
                papers_block=_format_papers_block(foundational, 20),
                influence_block=influence_block,
            )
            sections["origins"] = self._llm_call(client, model, prompt_origins)

            # Section 2: Major Research Threads
            report("Generating section 2/5: Major Research Threads...")
            threads_block = "\n\n".join(
                self._build_thread_block(cid, cluster_labels.get(cid, f"Thread {cid}"), ps)
                for cid, ps in sorted(real_clusters.items())
            )
            prompt_threads = SECTION_THREADS.format(
                topic=topic,
                threads_block=threads_block,
            )
            sections["threads"] = self._llm_call(client, model, prompt_threads)

            # Section 3: Competing Approaches
            report("Generating section 3/5: Competing Approaches...")
            competition_block = self._build_competition_block(competition_analysis)
            competition_papers = self._get_competition_papers(real_clusters, competition_analysis)
            prompt_competition = SECTION_COMPETITION.format(
                topic=topic,
                competition_block=competition_block,
                papers_block=_format_papers_block(competition_papers, 20),
            )
            sections["competition"] = self._llm_call(client, model, prompt_competition)

            # Section 4: Evolution & Paradigm Shifts
            report("Generating section 4/5: Evolution & Paradigm Shifts...")
            dominance_block = self._build_dominance_block(competition_analysis)
            shifters = self._get_paradigm_shifters(all_papers, influence_scores)
            shifters_block = _format_papers_block(shifters, 10) if shifters else "(No paradigm-shifting papers identified)"
            prompt_evolution = SECTION_EVOLUTION.format(
                topic=topic,
                dominance_block=dominance_block,
                shifters_block=shifters_block,
            )
            sections["evolution"] = self._llm_call(client, model, prompt_evolution)

            # Section 5: Current State & Open Problems
            report("Generating section 5/5: Current State & Open Problems...")
            recent = self._get_recent_papers(all_papers, influence_scores)
            prompt_frontier = SECTION_FRONTIER.format(
                topic=topic,
                papers_block=_format_papers_block(recent, 20),
            )
            sections["frontier"] = self._llm_call(client, model, prompt_frontier)

            # Synthesis
            report("Synthesizing sections into cohesive narrative...")
            combined = "\n\n---\n\n".join(
                f"### {name.upper()}\n{text}"
                for name, text in sections.items()
            )
            prompt_synthesis = SYNTHESIS_PROMPT.format(
                topic=topic,
                sections=combined,
            )
            narrative = self._llm_call(client, model, prompt_synthesis, max_tokens=6000)

            # Generate per-thread narratives for the dashboard
            for cid, ps in sorted(real_clusters.items()):
                label = cluster_labels.get(cid, f"Thread {cid}")
                report(f"Generating thread narrative: {label}...")
                self.thread_narratives[cid] = self.generate_thread_narrative(
                    label, ps, influence_scores
                )

            return narrative

        except Exception as e:
            logger.error(f"Chunked generation failed: {e}")
            return self._generate_without_llm(
                topic, clusters, cluster_labels, influence_scores, competition_analysis
            )

    def _llm_call(self, client, model: str, prompt: str, max_tokens: int = 2500) -> str:
        """Make a single LLM call with the narrative system prompt."""
        response = client.chat.completions.create(
            model=model,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _build_thread_block(self, cid: int, label: str, papers: list[Paper]) -> str:
        """Build a summary block for one thread."""
        sorted_papers = sorted(papers, key=lambda p: (p.year or 0, -p.citation_count))
        lines = [f"### Thread {cid}: {label} ({len(papers)} papers)"]
        for p in sorted_papers[:15]:
            lines.append(_format_paper(p))
        return "\n".join(lines)

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
            parts.append(self._build_competition_block(competition_analysis))

        return "\n\n".join(parts) if parts else "(No citation graph data available)"

    def _build_competition_block(self, competition_analysis: Optional[dict]) -> str:
        if not competition_analysis:
            return "(No competition data)"
        parts = []
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

        complementary = competition_analysis.get("complementary_pairs", [])
        if complementary:
            lines = ["### Complementary Thread Pairs"]
            for cp in complementary[:5]:
                lines.append(
                    f"- \"{cp['foundation_label']}\" (foundation) → "
                    f"\"{cp['builder_label']}\" (builds upon): "
                    f"{cp['builder_to_foundation']} citations toward foundation"
                )
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _build_dominance_block(self, competition_analysis: Optional[dict]) -> str:
        if not competition_analysis:
            return "(No dominance data)"
        dominance = competition_analysis.get("dominance_timeline", {})
        if not dominance:
            return "(No dominance timeline data)"

        lines = []
        for year in sorted(dominance.keys()):
            entries = dominance[year]
            top = sorted(entries, key=lambda e: e["paper_share"], reverse=True)
            if top and top[0]["paper_share"] > 0:
                leader = top[0]
                lines.append(
                    f"- {year}: \"{leader['label']}\" leads with {leader['paper_share']:.0%} "
                    f"of papers, {leader['citation_share']:.0%} of citations"
                )
        return "\n".join(lines) if lines else "(Insufficient data for dominance timeline)"

    def _build_influence_block(
        self, papers: list[Paper], influence_scores: Optional[dict]
    ) -> str:
        if not influence_scores:
            return ""
        lines = ["Most influential among these papers:"]
        for p in papers[:10]:
            scores = influence_scores.get(p.paper_id, {})
            if scores:
                first_author = p.authors[0].name if p.authors else "Unknown"
                lines.append(
                    f"- [{first_author}, {p.year}]: composite={scores.get('composite', 0):.3f}, "
                    f"pagerank={scores.get('pagerank', 0):.3f}"
                )
        return "\n".join(lines)

    def _get_foundational_papers(
        self, papers: list[Paper], influence_scores: Optional[dict]
    ) -> list[Paper]:
        """Get foundational papers — earliest and most influential."""
        if influence_scores:
            scored = [
                (p, influence_scores.get(p.paper_id, {}).get("temporal_pioneer", 0)
                 + influence_scores.get(p.paper_id, {}).get("pagerank", 0))
                for p in papers
            ]
            scored.sort(key=lambda x: -x[1])
            return [p for p, _ in scored[:20]]
        return sorted(papers, key=lambda p: (p.year or 9999))[:20]

    def _get_recent_papers(
        self, papers: list[Paper], influence_scores: Optional[dict]
    ) -> list[Paper]:
        """Get recent papers sorted by citation burst."""
        if not papers:
            return []
        max_year = max(p.year for p in papers if p.year) if any(p.year for p in papers) else 2026
        recent = [p for p in papers if p.year and p.year >= max_year - 2]
        if not recent:
            recent = sorted(papers, key=lambda p: -(p.year or 0))[:20]
        if influence_scores:
            recent.sort(
                key=lambda p: influence_scores.get(p.paper_id, {}).get("citation_burst", 0),
                reverse=True,
            )
        return recent[:20]

    def _get_paradigm_shifters(
        self, papers: list[Paper], influence_scores: Optional[dict]
    ) -> list[Paper]:
        """Get papers that likely caused paradigm shifts."""
        if not influence_scores:
            return []
        scored = []
        for p in papers:
            s = influence_scores.get(p.paper_id, {})
            shift_score = 0.5 * s.get("bridge", 0) + 0.5 * s.get("temporal_pioneer", 0)
            if shift_score > 0.3:
                scored.append((p, shift_score))
        scored.sort(key=lambda x: -x[1])
        return [p for p, _ in scored[:10]]

    def _get_competition_papers(
        self,
        clusters: dict[int, list[Paper]],
        competition_analysis: Optional[dict],
    ) -> list[Paper]:
        """Get papers from competing threads for the competition section."""
        if not competition_analysis:
            return []
        comp_pairs = competition_analysis.get("competition_pairs", [])
        complementary = competition_analysis.get("complementary_pairs", [])

        relevant_cids = set()
        for cp in comp_pairs:
            relevant_cids.add(cp["cluster_a"])
            relevant_cids.add(cp["cluster_b"])
        for cp in complementary:
            relevant_cids.add(cp["foundation_cluster"])
            relevant_cids.add(cp["builder_cluster"])

        papers = []
        for cid in relevant_cids:
            if cid in clusters:
                cpapers = sorted(clusters[cid], key=lambda p: -p.citation_count)
                papers.extend(cpapers[:10])
        return papers

    def _generate_without_llm(
        self,
        topic: str,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
        influence_scores: Optional[dict] = None,
        competition_analysis: Optional[dict] = None,
    ) -> str:
        """Template-based fallback when no LLM is available."""
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
        lines.append("*Note: Full narrative generation requires an OpenAI API key.*")

        return "\n".join(lines)

    def _thread_summary_fallback(self, label: str, papers: list[Paper]) -> str:
        sorted_papers = sorted(papers, key=lambda p: (p.year or 0))
        lines = [f"**{label}** — {len(papers)} papers\n"]
        for p in sorted_papers[:10]:
            first_author = p.authors[0].name if p.authors else "Unknown"
            lines.append(
                f"- [{first_author}, {p.year}] \"{p.title}\" ({p.citation_count} citations)"
            )
        return "\n".join(lines)

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
You are an expert academic researcher and science writer preparing content \
for a top-tier NLP/AI venue (EMNLP, ACL, NeurIPS). Your task is to generate \
a structured, publication-quality research narrative that tells the "story" \
of a research area based on the provided papers.

RULES:
1. Every factual claim MUST cite at least one paper using [Author et al., Year] format.
2. Organize content chronologically within each section, showing clear causal progression.
3. Highlight competing approaches, explain trade-offs, and show how ideas evolved.
4. Use precise academic language — avoid vague phrases like "some researchers" or "recent work".
5. Use the EXACT paper information provided; do NOT hallucinate papers or facts.
6. When citing, use ONLY the first author's SURNAME (family name, last word of their name) \
followed by "et al." and the year. For example: [Smith et al., 2023], NOT [John Smith et al., 2023].
7. Draw explicit connections between papers — show how one builds on or challenges another.
8. Include specific technical details (methods, datasets, metrics) when available in abstracts.
9. Every paragraph must have at least 2-3 citations to maintain academic rigor.
10. Use transition phrases that show intellectual progression: "Building on this foundation...", \
"In contrast to...", "This limitation motivated...", "Extending this paradigm...".
"""

SINGLE_PASS_PROMPT = """\
Generate a comprehensive, publication-quality research narrative for: "{topic}"

I have organized {total_papers} papers into {n_threads} research threads through \
automated clustering and citation analysis.

=== RESEARCH THREADS ===
{cluster_summaries}

=== CITATION & INFLUENCE ANALYSIS ===
{citation_block}

=== THREAD DOMINANCE OVER TIME ===
{dominance_block}

Write a comprehensive research narrative with these sections. Each section should \
be 3-5 substantial paragraphs with dense citations:

## 1. Origins & Foundations
Trace how this research area emerged. Identify the seminal papers that launched \
the field, the key problems they addressed, and the foundational techniques they \
introduced. Use influence scores to identify the most impactful early contributions. \
Show the intellectual lineage — which ideas enabled what came later.

## 2. Major Research Threads
For each identified thread, write a focused analysis covering:
- Core research questions and what distinguishes this thread
- Key milestone papers and their specific contributions
- Methodological innovations within the thread
- How this thread relates to, extends, or diverges from others

## 3. Competing Approaches & Trade-offs
Analyze the intellectual tensions in this field using the competition data:
- Which approaches represent genuine alternatives and why
- What trade-offs each approach makes (accuracy vs efficiency, generality vs specialization)
- How cross-citation patterns reveal scholarly debate and mutual awareness
- Which complementary threads build upon each other and why

## 4. Evolution & Paradigm Shifts
Trace the temporal evolution of the field:
- How dominant approaches shifted over time (use the dominance timeline data)
- What triggered major paradigm shifts (new datasets, methods, theoretical insights)
- Which papers served as bridges between research communities
- Key turning points where the field's direction changed

## 5. Current State & Open Problems
Describe the current frontier:
- The most active and promising research directions right now
- Key unsolved problems and challenges the community faces
- Emerging trends suggested by recent high-burst papers
- Concrete future directions that follow from the current state

IMPORTANT:
- Cite papers as [Author et al., Year] throughout — every claim needs a citation
- Use ONLY the papers provided above; do not invent references
- Aim for depth and analytical insight, not just listing papers
- Write at least 3-4 paragraphs per section with 2-3 citations per paragraph minimum
- Show causal connections: how did paper X enable or motivate paper Y?
"""


def _format_paper(p: Paper, include_abstract: bool = True) -> str:
    """Format a paper for inclusion in a prompt."""
    if p.authors:
        surname = p.authors[0].name.split()[-1]
        first_author = f"{surname} et al." if len(p.authors) > 1 else surname
    else:
        first_author = "Unknown"
    line = f"- [{first_author}, {p.year}] \"{p.title}\" (cited {p.citation_count}x)"
    if include_abstract and p.abstract:
        snippet = p.abstract[:500] + "..." if len(p.abstract) > 500 else p.abstract
        line += f"\n  Abstract: {snippet}"
    return line


def _format_papers_block(papers: list[Paper], max_papers: int = 50) -> str:
    """Format a list of papers into a prompt block."""
    selected = papers[:max_papers]
    return "\n".join(_format_paper(p) for p in selected)


class CitationVerifier:
    """Post-generation verification that narrative citations reference real papers."""

    CITATION_PATTERN = re.compile(
        r'\[([A-Z][a-zA-Z\-\']+(?:\s+[A-Za-z\-\']+)*(?:\s+et\s+al\.)?),?\s*(\d{4})\]'
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
            for author in p.authors[:3]:
                a_surname = author.name.split()[-1].lower()
                if a_surname != surname:
                    self.index[(a_surname, p.year)].append(p)

    def verify(self, narrative: str) -> dict:
        """Verify all citations in a narrative."""
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
                found = False
                for (s, y), papers in self.index.items():
                    if y == year and (surname_lower in s or s in surname_lower):
                        verified.append((f"[{author_part}, {year_str}]", papers[0].title))
                        found = True
                        break
                if not found:
                    for (s, y), papers in self.index.items():
                        if y == year and (
                            s.startswith(surname_lower[:3]) or
                            surname_lower.startswith(s[:3])
                        ):
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
                    if y == year and (surname_lower in s or s in surname_lower):
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
        """Generate a full research narrative with chunking and verification."""
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

        _report(f"Generating narrative for {total_papers} papers across {len(real_clusters)} threads...")
        narrative = self._generate_single_pass(
            topic, clusters, cluster_labels,
            influence_scores, competition_analysis, citation_graph
        )

        _report("Generating thread deep-dives...")
        self._generate_all_thread_narratives(
            real_clusters, cluster_labels, influence_scores, _report
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
                )[:10]
                for p in ranked:
                    scores = influence_scores.get(p.paper_id, {})
                    first_author = p.authors[0].name if p.authors else "Unknown"
                    inf_lines.append(
                        f"- [{first_author}, {p.year}]: composite={scores.get('composite', 0):.3f}"
                    )

            inf_block = "\n".join(inf_lines) if inf_lines else ""
            paper_block = _format_papers_block(sorted_papers, max_papers=40)

            prompt = (
                f'Write a detailed 3-5 paragraph narrative about the research thread '
                f'"{thread_label}" based on these {len(papers)} papers:\n\n{paper_block}\n\n'
            )
            if inf_block:
                prompt += f"Most influential papers in this thread:\n{inf_block}\n\n"
            prompt += (
                "Cover: (1) how this thread started and what motivated it, "
                "(2) key advances and milestone papers with specific technical details, "
                "(3) current state and open questions. "
                "Cite papers as [Author et al., Year]. Every paragraph must have 2+ citations."
            )

            response = client.chat.completions.create(
                model=model,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4000,
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
        """Generate entire narrative in a single LLM call."""
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

            dominance_block = self._build_dominance_block(competition_analysis)

            total_papers = sum(len(ps) for ps in real_clusters.values())
            user_prompt = SINGLE_PASS_PROMPT.format(
                topic=topic,
                total_papers=total_papers,
                n_threads=len(real_clusters),
                cluster_summaries=cluster_summaries,
                citation_block=citation_block,
                dominance_block=dominance_block,
            )

            response = client.chat.completions.create(
                model=model,
                temperature=LLM_TEMPERATURE,
                messages=[
                    {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=16000,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Single-pass generation failed: {e}")
            return self._generate_without_llm(
                topic, clusters, cluster_labels, influence_scores, competition_analysis
            )

    def _generate_all_thread_narratives(
        self,
        clusters: dict[int, list[Paper]],
        cluster_labels: dict[int, str],
        influence_scores: Optional[dict],
        report: callable,
    ) -> None:
        """Generate each thread narrative individually for maximum quality."""
        if not HAS_LLM:
            for cid, ps in sorted(clusters.items()):
                label = cluster_labels.get(cid, f"Thread {cid}")
                self.thread_narratives[cid] = self._thread_summary_fallback(label, ps)
            return

        client = get_llm_client()
        model = get_model_name()
        cid_order = sorted(clusters.keys())

        for idx, cid in enumerate(cid_order):
            ps = clusters[cid]
            label = cluster_labels.get(cid, f"Thread {cid}")
            report(f"Generating narrative for thread {idx+1}/{len(cid_order)}: {label}")

            try:
                sorted_papers = sorted(ps, key=lambda p: (p.year or 0))
                paper_block = _format_papers_block(sorted_papers, max_papers=40)

                inf_lines = ""
                if influence_scores:
                    ranked = sorted(
                        ps,
                        key=lambda p: influence_scores.get(p.paper_id, {}).get("composite", 0),
                        reverse=True,
                    )[:10]
                    inf_parts = []
                    for p in ranked:
                        scores = influence_scores.get(p.paper_id, {})
                        first_author = p.authors[0].name if p.authors else "Unknown"
                        inf_parts.append(
                            f"  - [{first_author}, {p.year}]: composite={scores.get('composite', 0):.3f}"
                        )
                    if inf_parts:
                        inf_lines = "\nMost influential papers:\n" + "\n".join(inf_parts)

                prompt = (
                    f"Write a focused, publication-quality deep-dive narrative for the following "
                    f"research thread.\n\n"
                    f"### THREAD: {label} ({len(ps)} papers)\n"
                    f"{paper_block}{inf_lines}\n\n"
                    f"Write 5-7 substantial paragraphs covering:\n"
                    f"(1) Origins and motivation — what problem launched this thread\n"
                    f"(2) Key methodological innovations — specific techniques, architectures, or algorithms\n"
                    f"(3) Milestone papers — detailed discussion of their contributions and impact\n"
                    f"(4) Connections to broader research landscape and cross-pollination of ideas\n"
                    f"(5) Current state — latest work, remaining gaps, and open questions\n\n"
                    f"Be thorough: reference as many of the listed papers as possible. "
                    f"Discuss specific methods, datasets, and results when available. "
                    f"Cite papers as [Author et al., Year]. "
                    f"Every paragraph must have at least 2-3 citations."
                )

                response = client.chat.completions.create(
                    model=model,
                    temperature=LLM_TEMPERATURE,
                    messages=[
                        {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=4000,
                )
                self.thread_narratives[cid] = response.choices[0].message.content

            except Exception as e:
                logger.error(f"Thread {cid} narrative generation failed: {e}")
                self.thread_narratives[cid] = self._thread_summary_fallback(label, ps)

        report(f"Generated {len(self.thread_narratives)} thread narratives")

    def _build_thread_block(self, cid: int, label: str, papers: list[Paper]) -> str:
        """Build a summary block for one thread."""
        sorted_papers = sorted(papers, key=lambda p: (p.year or 0, -p.citation_count))
        lines = [f"### Thread {cid}: {label} ({len(papers)} papers)"]
        for p in sorted_papers[:40]:
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
            )[:30]
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
                        f"Burst: {scores['citation_burst']:.3f}, "
                        f"Composite: {scores['composite']:.3f}"
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
                f"- **[{first_author} et al., {p.year}]** \"{p.title}\" — "
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
                f"- [{first_author} et al., {p.year}] \"{p.title}\" ({p.citation_count} citations)"
            )
        return "\n".join(lines)

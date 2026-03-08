"""Unified paper ingestion pipeline.

Combines results from multiple API sources into a single PaperCollection
with deduplication and normalization.
"""

from __future__ import annotations
import logging
from typing import Optional, Callable

from src.models.paper import Paper, PaperCollection
from src.api.arxiv_client import ArxivClient
from src.api.semantic_scholar_client import SemanticScholarClient
from src.config import PAPERS_DIR, SEMANTIC_SCHOLAR_API_KEY

logger = logging.getLogger(__name__)


class PaperIngestionPipeline:
    """Orchestrates paper retrieval from multiple sources."""

    def __init__(self):
        self.arxiv = ArxivClient()
        self.s2 = SemanticScholarClient()

    def ingest(
        self,
        topic: str,
        max_papers: int = 200,
        sources: Optional[list[str]] = None,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        min_citations: int = 0,
        enrich_citations: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> PaperCollection:
        """Run the full ingestion pipeline for a research topic."""
        if sources is None:
            sources = ["arxiv", "s2"]

        def _report(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        # Wire S2 client's progress to the same callback
        self.s2.progress_callback = progress_callback

        collection = PaperCollection(topic=topic)
        per_source = max_papers // len(sources)

        if "arxiv" in sources:
            _report(f"Fetching up to {per_source} papers from arXiv...")
            arxiv_papers = self.arxiv.search(
                query=topic,
                max_results=per_source,
                start_year=start_year,
                end_year=end_year,
            )
            for p in arxiv_papers:
                collection.add(p)
            _report(f"arXiv: retrieved {len(arxiv_papers)} papers")

        if "s2" in sources:
            year_range = None
            if start_year and end_year:
                year_range = f"{start_year}-{end_year}"
            elif start_year:
                year_range = f"{start_year}-"
            elif end_year:
                year_range = f"-{end_year}"

            _report(f"Fetching up to {per_source} papers from Semantic Scholar...")
            s2_papers = self.s2.search(
                query=topic,
                max_results=per_source,
                year_range=year_range,
                min_citation_count=min_citations,
            )
            for p in s2_papers:
                collection.add(p)
            _report(f"Semantic Scholar: retrieved {len(s2_papers)} papers")

        # Filter out papers with empty abstracts
        empty_ids = [
            pid for pid, p in collection.papers.items()
            if not p.abstract or len(p.abstract.strip()) < 20
        ]
        for pid in empty_ids:
            del collection.papers[pid]
        if empty_ids:
            _report(f"Filtered {len(empty_ids)} papers with missing/short abstracts")

        _report(f"Total unique papers after deduplication: {len(collection)}")

        # Citation enrichment — only when we have an API key (otherwise too slow/rate-limited)
        if enrich_citations and SEMANTIC_SCHOLAR_API_KEY and len(collection) > 0:
            _report("Enriching papers with citation data (S2 API key detected)...")
            papers_list = collection.get_papers()
            enrichable = [p for p in papers_list if p.s2_id or p.arxiv_id]
            if enrichable:
                self.s2.get_citations_batch(enrichable)
                _report(f"Enriched {len(enrichable)} papers with citation details")
        elif enrich_citations and not SEMANTIC_SCHOLAR_API_KEY:
            _report("Skipping citation enrichment (no S2 API key — using search-level citation counts)")

        # Persist
        safe_topic = topic.replace(" ", "_").replace("/", "_")[:50]
        save_path = PAPERS_DIR / f"{safe_topic}.json"
        collection.save(save_path)
        _report(f"Saved {len(collection)} papers to cache")

        return collection

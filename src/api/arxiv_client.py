"""arXiv API client for paper retrieval.

Uses the arXiv Atom feed API with rate limiting and pagination.
Docs: https://info.arxiv.org/help/api/index.html
"""

from __future__ import annotations
import time
import logging
import xmltodict
import requests
from typing import Optional

from src.models.paper import Paper, Author
from src.config import ARXIV_RATE_LIMIT

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
MAX_RESULTS_PER_PAGE = 100


class ArxivClient:
    """Fetches papers from the arXiv API."""

    def __init__(self, rate_limit: float = ARXIV_RATE_LIMIT):
        self.rate_limit = rate_limit
        self._last_request_time = 0.0

    def _wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def search(
        self,
        query: str,
        max_results: int = 100,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        categories: Optional[list[str]] = None,
    ) -> list[Paper]:
        """Search arXiv for papers matching a query.

        Args:
            query: Search query string (supports arXiv query syntax).
            max_results: Maximum number of papers to retrieve.
            sort_by: Sort criterion – "relevance", "lastUpdatedDate", "submittedDate".
            sort_order: "ascending" or "descending".
            start_year: Filter papers published on or after this year.
            end_year: Filter papers published on or before this year.
            categories: arXiv category filters (e.g., ["cs.CL", "cs.AI"]).

        Returns:
            List of Paper objects normalized from arXiv results.
        """
        search_query = self._build_query(query, categories)
        papers: list[Paper] = []
        start = 0

        while start < max_results:
            batch_size = min(MAX_RESULTS_PER_PAGE, max_results - start)
            batch = self._fetch_page(search_query, start, batch_size, sort_by, sort_order)
            if not batch:
                break

            for paper in batch:
                if start_year and paper.year and paper.year < start_year:
                    continue
                if end_year and paper.year and paper.year > end_year:
                    continue
                papers.append(paper)

            start += batch_size
            if len(batch) < batch_size:
                break

        logger.info(f"arXiv: retrieved {len(papers)} papers for query '{query}'")
        return papers

    def _build_query(self, query: str, categories: Optional[list[str]] = None) -> str:
        # Split query into individual terms and AND them together
        # so "RAG based models in healthcare" becomes
        # all:RAG AND all:based AND all:models AND all:healthcare
        stop_words = {"in", "on", "of", "the", "a", "an", "for", "and", "or", "to", "with"}
        terms = [t.strip() for t in query.split() if t.strip().lower() not in stop_words]
        if terms:
            term_query = " AND ".join(f"all:{t}" for t in terms)
        else:
            term_query = f"all:{query}"

        parts = [term_query]
        if categories:
            cat_query = " OR ".join(f"cat:{c}" for c in categories)
            parts.append(f"({cat_query})")
        return " AND ".join(parts)

    def _fetch_page(
        self, query: str, start: int, max_results: int,
        sort_by: str, sort_order: str
    ) -> list[Paper]:
        self._wait()
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        try:
            resp = requests.get(ARXIV_API_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"arXiv API error: {e}")
            return []

        return self._parse_response(resp.text)

    def _parse_response(self, xml_text: str) -> list[Paper]:
        parsed = xmltodict.parse(xml_text)
        feed = parsed.get("feed", {})
        entries = feed.get("entry", [])

        if isinstance(entries, dict):
            entries = [entries]

        papers = []
        for entry in entries:
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)
        return papers

    def _parse_entry(self, entry: dict) -> Optional[Paper]:
        try:
            raw_id = entry.get("id", "")
            # arXiv IDs come as URLs like http://arxiv.org/abs/2201.00978v1
            arxiv_id = raw_id.split("/")[-1].split("v")[0]
            if not arxiv_id or arxiv_id == raw_id:
                return None

            title = entry.get("title", "").replace("\n", " ").strip()
            abstract = entry.get("summary", "").replace("\n", " ").strip()

            # Parse authors
            authors_raw = entry.get("author", [])
            if isinstance(authors_raw, dict):
                authors_raw = [authors_raw]
            authors = [
                Author(
                    name=a.get("name", "Unknown"),
                    affiliation=a.get("arxiv:affiliation", {}).get("#text")
                    if isinstance(a.get("arxiv:affiliation"), dict) else None,
                )
                for a in authors_raw
            ]

            # Parse date
            published = entry.get("published", "")
            year = int(published[:4]) if len(published) >= 4 else None
            month = int(published[5:7]) if len(published) >= 7 else None

            # Categories
            cats_raw = entry.get("category", [])
            if isinstance(cats_raw, dict):
                cats_raw = [cats_raw]
            categories = [c.get("@term", "") for c in cats_raw if c.get("@term")]

            # URL
            links = entry.get("link", [])
            if isinstance(links, dict):
                links = [links]
            url = None
            for link in links:
                if link.get("@type") == "text/html":
                    url = link.get("@href")
                    break
            if not url:
                url = f"https://arxiv.org/abs/{arxiv_id}"

            # DOI
            doi = entry.get("arxiv:doi", {})
            if isinstance(doi, dict):
                doi = doi.get("#text")

            return Paper(
                paper_id=f"arxiv:{arxiv_id}",
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                month=month,
                doi=doi if isinstance(doi, str) else None,
                url=url,
                source="arxiv",
                arxiv_id=arxiv_id,
                categories=categories,
            )
        except Exception as e:
            logger.warning(f"Failed to parse arXiv entry: {e}")
            return None

"""Semantic Scholar API client for paper retrieval and citation extraction.

Uses the Semantic Scholar Academic Graph API.
Docs: https://api.semanticscholar.org/api-docs/graph
"""

from __future__ import annotations
import time
import logging
import requests
from typing import Optional

from src.models.paper import Paper, Author
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.config import (
    SEMANTIC_SCHOLAR_API_KEY,
    S2_RATE_LIMIT,
    S2_RATE_LIMIT_NO_KEY,
)

logger = logging.getLogger(__name__)

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_SEARCH_URL = f"{S2_API_BASE}/paper/search"
S2_PAPER_URL = f"{S2_API_BASE}/paper"
S2_BATCH_URL = f"{S2_API_BASE}/paper/batch"
S2_PAPER_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,"
    "citationCount,influentialCitationCount,"
    "authors,references,citations,fieldsOfStudy"
)
S2_SEARCH_FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,"
    "citationCount,influentialCitationCount,authors,fieldsOfStudy"
)


class SemanticScholarClient:
    """Fetches papers and citations from the Semantic Scholar API."""

    def __init__(self, progress_callback=None):
        self.api_key = SEMANTIC_SCHOLAR_API_KEY
        self.rate_limit = S2_RATE_LIMIT if self.api_key else S2_RATE_LIMIT_NO_KEY
        self._last_request_time = 0.0
        self.progress_callback = progress_callback
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["x-api-key"] = self.api_key

    def _report(self, msg: str):
        logger.info(msg)
        if self.progress_callback:
            self.progress_callback(msg)

    def _wait(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _request_with_retry(self, url: str, params: dict) -> Optional[dict]:
        """Make a GET request with retry on 429 rate limit errors.

        Non-retryable status codes (404, 400, 403) are skipped immediately.
        """
        max_retries = 10
        delay = 5
        non_retryable = {400, 403, 404, 405, 410}
        resp = None
        for attempt in range(max_retries):
            self._wait()
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code in non_retryable:
                    logger.debug(f"S2 returned {resp.status_code} — skipping (not retryable)")
                    return None
                if resp.status_code == 429:
                    self._report(f"S2 rate limited (429). Retrying in {delay}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if resp is not None and resp.status_code in non_retryable:
                    return None
                if resp is not None and resp.status_code == 429:
                    self._report(f"S2 rate limited (429). Retrying in {delay}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                self._report(f"S2 request error ({type(e).__name__}). Retrying in {delay}s (attempt {attempt+1}/{max_retries})...")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                return None
        self._report(f"S2 request failed after {max_retries} attempts — skipping")
        return None

    def search(
        self,
        query: str,
        max_results: int = 100,
        year_range: Optional[str] = None,
        fields_of_study: Optional[list[str]] = None,
        min_citation_count: int = 0,
    ) -> list[Paper]:
        """Search for papers by query string.

        Args:
            query: Natural language search query.
            max_results: Maximum papers to return (API max 100 per page).
            year_range: Year filter, e.g. "2018-2024" or "2020-".
            fields_of_study: Filter by field, e.g. ["Computer Science"].
            min_citation_count: Minimum citation count filter.

        Returns:
            List of Paper objects.
        """
        papers: list[Paper] = []
        offset = 0
        limit = min(100, max_results)

        while offset < max_results:
            params: dict = {
                "query": query,
                "offset": offset,
                "limit": limit,
                "fields": S2_SEARCH_FIELDS,
            }
            if year_range:
                params["year"] = year_range
            if fields_of_study:
                params["fieldsOfStudy"] = ",".join(fields_of_study)
            if min_citation_count:
                params["minCitationCount"] = min_citation_count

            data = self._request_with_retry(S2_SEARCH_URL, params)
            if not data:
                break

            results = data.get("data", [])
            if not results:
                break

            for item in results:
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)

            offset += limit
            total = data.get("total", 0)
            if offset >= total:
                break

        logger.info(f"Semantic Scholar: retrieved {len(papers)} papers for '{query}'")
        return papers

    def get_paper_details(self, paper_id: str) -> Optional[Paper]:
        """Get full details for a single paper including references and citations."""
        url = f"{S2_PAPER_URL}/{paper_id}"
        params = {"fields": S2_PAPER_FIELDS}

        data = self._request_with_retry(url, params)
        if not data:
            return None

        return self._parse_paper_with_citations(data)

    def get_citations_batch(self, papers: list[Paper], max_per_paper: int = 100) -> list[Paper]:
        """Enrich papers with references and citation data using the S2 batch API.

        Uses POST /paper/batch to fetch up to 500 papers per request,
        dramatically faster than individual GET calls.
        """
        # Build lookup: s2_id -> paper (for matching responses back)
        id_to_paper: dict[str, Paper] = {}
        batch_ids: list[str] = []
        for p in papers:
            lookup_id = p.s2_id if p.s2_id else None
            if lookup_id:
                id_to_paper[lookup_id] = p
                batch_ids.append(lookup_id)

        if not batch_ids:
            self._report("No S2 IDs available for batch enrichment — skipping")
            return papers

        BATCH_SIZE = 500
        total_batches = (len(batch_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        enriched_count = 0

        for batch_num in range(total_batches):
            start = batch_num * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(batch_ids))
            chunk = batch_ids[start:end]

            self._report(
                f"Batch enrichment {batch_num + 1}/{total_batches}: "
                f"fetching {len(chunk)} papers ({start + 1}–{end} of {len(batch_ids)})..."
            )

            data = self._post_batch(chunk)
            if not data:
                self._report(f"Batch {batch_num + 1} failed — skipping this chunk")
                continue

            for item in data:
                if not item:
                    continue
                s2_id = item.get("paperId")
                if not s2_id or s2_id not in id_to_paper:
                    continue

                paper = id_to_paper[s2_id]

                refs_raw = item.get("references") or []
                paper.references = [r["paperId"] for r in refs_raw if r.get("paperId")]

                cites_raw = item.get("citations") or []
                paper.cited_by = [c["paperId"] for c in cites_raw if c.get("paperId")][:max_per_paper]

                paper.citation_count = item.get("citationCount", paper.citation_count) or paper.citation_count
                paper.influential_citation_count = (
                    item.get("influentialCitationCount", paper.influential_citation_count)
                    or paper.influential_citation_count
                )
                enriched_count += 1

        self._report(f"Citation enrichment complete: {enriched_count}/{len(papers)} papers enriched via batch API")
        return papers

    def _post_batch(self, paper_ids: list[str]) -> Optional[list[dict]]:
        """POST to /paper/batch to fetch multiple papers in one request."""
        max_retries = 5
        delay = 5
        non_retryable = {400, 403, 404, 405, 410}
        resp = None

        for attempt in range(max_retries):
            self._wait()
            try:
                resp = self.session.post(
                    S2_BATCH_URL,
                    params={"fields": S2_PAPER_FIELDS},
                    json={"ids": paper_ids},
                    timeout=60,
                )
                if resp.status_code in non_retryable:
                    logger.debug(f"S2 batch returned {resp.status_code} — skipping")
                    return None
                if resp.status_code == 429:
                    self._report(f"S2 batch rate limited (429). Retrying in {delay}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if resp is not None and resp.status_code in non_retryable:
                    return None
                if resp is not None and resp.status_code == 429:
                    self._report(f"S2 batch rate limited (429). Retrying in {delay}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                    continue
                self._report(f"S2 batch request error ({type(e).__name__}). Retrying in {delay}s (attempt {attempt + 1}/{max_retries})...")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                return None
        self._report(f"S2 batch request failed after {max_retries} attempts — skipping")
        return None

    def _parse_paper(self, item: dict) -> Optional[Paper]:
        try:
            s2_id = item.get("paperId", "")
            if not s2_id:
                return None

            ext_ids = item.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv")
            doi = ext_ids.get("DOI")

            paper_id = f"arxiv:{arxiv_id}" if arxiv_id else f"s2:{s2_id}"

            authors = [
                Author(
                    name=a.get("name", "Unknown"),
                    author_id=a.get("authorId"),
                )
                for a in (item.get("authors") or [])
            ]

            categories = item.get("fieldsOfStudy") or []

            return Paper(
                paper_id=paper_id,
                title=(item.get("title") or "").strip(),
                abstract=(item.get("abstract") or "").strip(),
                authors=authors,
                year=item.get("year"),
                venue=item.get("venue"),
                doi=doi,
                url=f"https://www.semanticscholar.org/paper/{s2_id}",
                source="s2",
                arxiv_id=arxiv_id,
                s2_id=s2_id,
                citation_count=item.get("citationCount", 0) or 0,
                influential_citation_count=item.get("influentialCitationCount", 0) or 0,
                categories=categories,
            )
        except Exception as e:
            logger.warning(f"Failed to parse S2 paper: {e}")
            return None

    def _parse_paper_with_citations(self, item: dict) -> Optional[Paper]:
        paper = self._parse_paper(item)
        if not paper:
            return None

        refs_raw = item.get("references") or []
        paper.references = [
            r["paperId"] for r in refs_raw
            if r.get("paperId")
        ]

        cites_raw = item.get("citations") or []
        paper.cited_by = [
            c["paperId"] for c in cites_raw
            if c.get("paperId")
        ]

        return paper

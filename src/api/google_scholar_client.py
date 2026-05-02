"""Google Scholar client using the scholarly2 library.

Provides paper search and citation retrieval from Google Scholar
as a supplementary data source for the ResearchNarrative pipeline.

NOTE: Google Scholar rate-limits aggressively. This client includes
built-in delays and will gracefully degrade if blocked.
"""

from __future__ import annotations
import logging
import time
from typing import Optional, Callable

from src.models.paper import Paper, Author

logger = logging.getLogger(__name__)

GS_RATE_LIMIT = 3.0


class GoogleScholarClient:
    """Retrieves papers from Google Scholar via scholarly2."""

    def __init__(self):
        self._scholarly = None
        self.progress_callback: Optional[Callable[[str], None]] = None

    def _get_scholarly(self):
        if self._scholarly is None:
            try:
                from scholarly2 import scholarly
                self._scholarly = scholarly
                logger.info("Google Scholar client initialized (scholarly2)")
            except ImportError:
                try:
                    from scholarly import scholarly
                    self._scholarly = scholarly
                    logger.info("Google Scholar client initialized (scholarly)")
                except ImportError:
                    logger.error(
                        "Neither scholarly2 nor scholarly is installed. "
                        "Install with: pip install scholarly2"
                    )
                    raise
        return self._scholarly

    def _report(self, msg: str):
        logger.info(msg)
        if self.progress_callback:
            self.progress_callback(msg)

    def search(
        self,
        query: str,
        max_results: int = 50,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> list[Paper]:
        """Search Google Scholar for papers matching a query."""
        scholarly = self._get_scholarly()
        papers = []

        try:
            self._report(f"Google Scholar: searching for '{query}'...")
            search_iter = scholarly.search_pubs(
                query,
                year_low=start_year,
                year_high=end_year,
            )

            for i in range(max_results):
                try:
                    result = next(search_iter)
                except StopIteration:
                    break

                paper = self._parse_result(result)
                if paper:
                    papers.append(paper)

                if (i + 1) % 10 == 0:
                    self._report(f"Google Scholar: fetched {i + 1}/{max_results} papers...")
                    time.sleep(GS_RATE_LIMIT)

            self._report(f"Google Scholar: retrieved {len(papers)} papers")

        except Exception as e:
            self._report(f"Google Scholar search error: {e}")
            logger.error(f"Google Scholar search failed: {e}")

        return papers

    def _parse_result(self, result: dict) -> Optional[Paper]:
        """Convert a scholarly result dict into a Paper object."""
        try:
            bib = result.get("bib", {})
            title = bib.get("title", "").strip()
            if not title:
                return None

            abstract = bib.get("abstract", "")
            year = bib.get("pub_year")
            if year:
                try:
                    year = int(year)
                except (ValueError, TypeError):
                    year = None

            authors = []
            for name in bib.get("author", []):
                if isinstance(name, str):
                    authors.append(Author(name=name.strip()))

            pub_url = result.get("pub_url") or result.get("eprint_url") or ""
            num_citations = result.get("num_citations", 0) or 0

            gs_id = result.get("author_pub_id", "") or result.get("url_scholarbib", "")
            paper_id = f"gs:{gs_id}" if gs_id else f"gs:{title[:60].replace(' ', '_')}"

            venue = bib.get("venue", "") or bib.get("journal", "") or bib.get("conference", "")

            return Paper(
                paper_id=paper_id,
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                venue=venue,
                url=pub_url,
                source="google_scholar",
                citation_count=int(num_citations),
            )

        except Exception as e:
            logger.warning(f"Failed to parse Google Scholar result: {e}")
            return None

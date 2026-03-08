"""Unified paper data model for ResearchNarrative.

All API sources (arXiv, Semantic Scholar, OpenAlex) are normalized into this schema.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class Author:
    name: str
    author_id: Optional[str] = None
    affiliation: Optional[str] = None


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: list[Author] = field(default_factory=list)
    year: Optional[int] = None
    month: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None

    # Source tracking
    source: str = ""                       # "arxiv", "s2", "openalex"
    arxiv_id: Optional[str] = None
    s2_id: Optional[str] = None
    openalex_id: Optional[str] = None

    # Citation info
    citation_count: int = 0
    influential_citation_count: int = 0
    references: list[str] = field(default_factory=list)    # paper_ids this cites
    cited_by: list[str] = field(default_factory=list)      # paper_ids that cite this

    # Categories / fields
    categories: list[str] = field(default_factory=list)

    # Embedding (populated later)
    embedding: Optional[list[float]] = None

    # Cluster assignment (populated later)
    cluster_id: int = -1
    cluster_label: str = ""

    def embedding_text(self) -> str:
        """Text used for generating embeddings (title + abstract)."""
        parts = []
        if self.title:
            parts.append(self.title)
        if self.abstract:
            parts.append(self.abstract)
        return " ".join(parts)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Don't persist large embedding vectors in JSON
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Paper:
        authors_raw = d.pop("authors", [])
        authors = [
            Author(**a) if isinstance(a, dict) else a
            for a in authors_raw
        ]
        d.pop("embedding", None)
        return cls(authors=authors, **d)


class PaperCollection:
    """Container for a set of papers with persistence."""

    def __init__(self, topic: str = ""):
        self.topic = topic
        self.papers: dict[str, Paper] = {}

    def add(self, paper: Paper) -> None:
        existing = self.papers.get(paper.paper_id)
        if existing:
            self._merge(existing, paper)
        else:
            self.papers[paper.paper_id] = paper

    def _merge(self, existing: Paper, new: Paper) -> None:
        """Merge data from a new source into an existing paper record."""
        if not existing.abstract and new.abstract:
            existing.abstract = new.abstract
        if new.citation_count > existing.citation_count:
            existing.citation_count = new.citation_count
        if new.influential_citation_count > existing.influential_citation_count:
            existing.influential_citation_count = new.influential_citation_count
        if new.references:
            existing.references = list(set(existing.references + new.references))
        if new.cited_by:
            existing.cited_by = list(set(existing.cited_by + new.cited_by))
        if new.s2_id and not existing.s2_id:
            existing.s2_id = new.s2_id
        if new.arxiv_id and not existing.arxiv_id:
            existing.arxiv_id = new.arxiv_id
        if new.categories:
            existing.categories = list(set(existing.categories + new.categories))

    def get_papers(self) -> list[Paper]:
        return list(self.papers.values())

    def __len__(self) -> int:
        return len(self.papers)

    def save(self, path: Path) -> None:
        data = {
            "topic": self.topic,
            "count": len(self.papers),
            "papers": [p.to_dict() for p in self.papers.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> PaperCollection:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        collection = cls(topic=data.get("topic", ""))
        for p in data.get("papers", []):
            collection.add(Paper.from_dict(p))
        return collection

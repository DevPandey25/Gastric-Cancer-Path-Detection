"""
Gastric Cancer Scientific Literature Retrieval and Verification Engine.
GastricPath AI — Phase 6: Evidence Retrieval and Literature Engine.

Interfaces with the verified 32-paper database (master_paper_database.csv)
and authoritative clinical guideline repositories. Literature cutoff: November 12, 2026.
"""

import csv
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parents[3]
MASTER_PAPER_CSV = BASE_DIR / "research" / "literature_review" / "master_paper_database.csv"


class LiteratureSearchEngine:
    """Retrieves verified peer-reviewed literature records from local database."""

    def __init__(self, csv_path: Path = MASTER_PAPER_CSV) -> None:
        self.csv_path = csv_path
        self.papers: list[dict[str, str]] = []
        self._load_papers()

    def _load_papers(self) -> None:
        if not self.csv_path.exists():
            return
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.papers.append(dict(row))

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Searches paper database across title, authors, task, and keywords."""
        q = query.lower().strip()
        results = []
        for paper in self.papers:
            searchable = (
                f"{paper.get('Title', '')} {paper.get('Authors', '')} "
                f"{paper.get('Task', '')} {paper.get('Modality_or_Dataset', '')} "
                f"{paper.get('Key_Relevance_to_GastricPath', '')}"
            ).lower()
            if q in searchable:
                results.append(paper)
            if len(results) >= limit:
                break
        return results

    def get_by_id(self, paper_id: str) -> Optional[dict[str, Any]]:
        """Fetches a specific verified paper record by Paper_ID (e.g., P01, P31)."""
        for paper in self.papers:
            if paper.get("Paper_ID") == paper_id:
                return paper
        return None

    def get_curated_reference_list(self) -> list[dict[str, str]]:
        """Returns verified reference citations formatted in Vancouver style."""
        formatted = []
        for p in self.papers:
            authors = p.get("Authors", "Unknown")
            title = p.get("Title", "Untitled")
            venue = p.get("Venue", "")
            year = p.get("Year", "")
            doi = p.get("DOI", "")
            citation = f"{authors}. {title}. {venue}. {year}; DOI: {doi}"
            formatted.append({
                "id": p.get("Paper_ID", ""),
                "citation": citation,
                "doi": doi,
                "title": title
            })
        return formatted


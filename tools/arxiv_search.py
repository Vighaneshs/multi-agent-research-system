"""arXiv paper search and download tools."""

from typing import List, Dict
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class ArxivSearchInput(BaseModel):
    query: str = Field(description="Search query for academic papers")
    max_results: int = Field(default=5, description="Maximum papers to return")
    sort_by: str = Field(default="relevance", description="relevance | submitted | updated")


@tool("arxiv_search", args_schema=ArxivSearchInput)
def arxiv_search_tool(
    query: str,
    max_results: int = 5,
    sort_by: str = "relevance",
) -> List[Dict]:
    """Search arXiv for papers matching the query."""
    try:
        import arxiv

        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "submitted": arxiv.SortCriterion.SubmittedDate,
            "updated": arxiv.SortCriterion.LastUpdatedDate,
        }

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance),
        )

        results = []
        for paper in search.results():
            results.append({
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "abstract": paper.summary,
                "pdf_url": paper.pdf_url,
                "arxiv_url": paper.entry_id,
                "published": paper.published.isoformat(),
                "updated": paper.updated.isoformat(),
                "categories": paper.categories,
                "primary_category": paper.primary_category,
            })
        return results

    except ImportError:
        return [{"error": "arxiv package not installed", "title": "Installation Required",
                 "abstract": "pip install arxiv"}]
    except Exception as e:
        return [{"error": str(e), "title": "Search Failed",
                 "abstract": f"Failed to search arxiv for: {query}"}]


@tool("arxiv_download")
def arxiv_download_tool(arxiv_id: str, output_dir: str = "./papers") -> Dict:
    """Download a paper PDF from arXiv by ID."""
    try:
        import arxiv
        import os

        os.makedirs(output_dir, exist_ok=True)
        paper = next(arxiv.Search(id_list=[arxiv_id]).results())
        filepath = paper.download_pdf(dirpath=output_dir)
        return {"success": True, "filepath": filepath, "title": paper.title}

    except Exception as e:
        return {"success": False, "error": str(e)}

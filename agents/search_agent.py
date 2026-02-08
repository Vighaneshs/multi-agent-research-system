"""Search agent — retrieves info from web and academic sources."""

from typing import Dict, List
from tavily import TavilyClient

from agents.base_agent import BaseAgent
from config import settings


class SearchAgent(BaseAgent):

    def __init__(self):
        self.tavily_client = None
        if settings.tavily_api_key:
            self.tavily_client = TavilyClient(api_key=settings.tavily_api_key)
        super().__init__(tools=[])

    @property
    def system_prompt(self) -> str:
        return """You are a Research Search Agent. Your job is to find relevant, 
accurate information from various sources.

When searching:
1. Formulate precise search queries
2. Look for authoritative sources (academic papers, official docs, reputable sites)
3. Verify information across multiple sources when possible
4. Extract key facts and their sources
5. Note the publication date for time-sensitive information

For each piece of information, always note:
- The source URL or citation
- The publication/update date if available
- Confidence level based on source reliability

Always prioritize accuracy over quantity."""

    def _search_web(self, query: str) -> List[Dict]:
        if not self.tavily_client:
            print("Web search not configured (missing TAVILY_API_KEY)")
            return self._mock_search_results(query)

        try:
            response = self.tavily_client.search(query=query, max_results=5)
            return [
                {
                    "source": r.get("url", "unknown"),
                    "title": r.get("title", "Untitled"),
                    "content": r.get("content", ""),
                    "relevance_score": r.get("score", 0.8),
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            print(f"Web search error: {e}")
            return []

    def _search_arxiv(self, query: str) -> List[Dict]:
        try:
            import arxiv

            search = arxiv.Search(
                query=query, max_results=5, sort_by=arxiv.SortCriterion.Relevance
            )
            results = []
            for paper in search.results():
                results.append({
                    "source": paper.pdf_url,
                    "title": paper.title,
                    "content": paper.summary,
                    "relevance_score": 0.9,
                    "authors": [a.name for a in paper.authors],
                    "published": paper.published.isoformat(),
                })
            return results
        except ImportError:
            print("arxiv package not installed — skipping academic search")
            return []
        except Exception as e:
            print(f"Arxiv search error: {e}")
            return []

    def _mock_search_results(self, query: str) -> List[Dict]:
        """Placeholder results for local testing without API keys."""
        return [
            {
                "source": "https://example.com/article1",
                "title": f"Understanding {query[:30]}...",
                "content": f"Mock result for: {query}",
                "relevance_score": 0.85,
            },
            {
                "source": "https://example.com/article2",
                "title": f"Deep Dive into {query[:30]}...",
                "content": "Another mock result covering the topic.",
                "relevance_score": 0.75,
            },
        ]

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        seen = set()
        out = []
        for r in results:
            src = r.get("source", "")
            if src not in seen:
                seen.add(src)
                out.append(r)
        return out

    def _rank_results(self, results: List[Dict]) -> List[Dict]:
        return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)

    def run(self, query: str, include_arxiv: bool = True) -> List[Dict]:
        print(f"Searching for: {query[:50]}...")

        all_results = self._search_web(query)
        print(f"  {len(all_results)} web results")

        if include_arxiv:
            arxiv_results = self._search_arxiv(query)
            all_results.extend(arxiv_results)
            print(f"  {len(arxiv_results)} academic papers")

        unique = self._deduplicate_results(all_results)
        ranked = self._rank_results(unique)
        print(f"  {len(ranked)} total unique results")
        return ranked

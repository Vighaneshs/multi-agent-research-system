"""Web search tools (Tavily + DuckDuckGo fallback)."""

from typing import List, Dict
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results")


@tool("web_search", args_schema=WebSearchInput)
def web_search_tool(query: str, max_results: int = 5) -> List[Dict]:
    """Search the web via Tavily and return a list of results."""
    from tavily import TavilyClient
    from config import settings

    if not settings.tavily_api_key:
        return [{
            "title": "Mock Result",
            "url": "https://example.com",
            "content": f"Mock search result for: {query}",
        }]

    client = TavilyClient(api_key=settings.tavily_api_key)
    resp = client.search(query=query, max_results=max_results)
    return resp.get("results", [])


class DuckDuckGoSearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results")


@tool("ddg_search", args_schema=DuckDuckGoSearchInput)
def duckduckgo_search_tool(query: str, max_results: int = 5) -> List[Dict]:
    """Fallback web search using DuckDuckGo (no API key needed)."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", ""),
                }
                for r in results
            ]
    except Exception as e:
        return [{
            "error": str(e),
            "title": "Search Failed",
            "url": "",
            "content": f"Failed to search for: {query}",
        }]

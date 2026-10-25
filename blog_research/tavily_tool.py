"""ADK function tool that searches Tavily for the web researcher."""

import asyncio
from dataclasses import asdict

from .tavily_client import TavilySearchClient

_SEARCH_CLIENT = TavilySearchClient()


async def tavily_search(
    query: str,
) -> dict[str, object]:
    """Search Tavily and return source titles, URLs, and excerpts."""
    search_query = query.strip()
    if not search_query:
        raise ValueError("A non-empty search query is required.")

    results = await asyncio.to_thread(_SEARCH_CLIENT.search, search_query, 5)
    return {
        "query": search_query,
        "sources": [asdict(result) for result in results],
    }

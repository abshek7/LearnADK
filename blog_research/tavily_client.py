"""Tavily Search API client."""

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class TavilySearchResult:
    """A source returned by Tavily Search."""

    title: str
    url: str
    content: str
    score: float | None = None
    published_date: str | None = None


class TavilySearchClient:
    """Search the web with the Tavily Search API."""

    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[TavilySearchResult]:
        """Return up to `limit` web sources matching a non-empty query."""
        if not query.strip():
            raise ValueError("A non-empty Tavily search query is required.")
        if not 1 <= limit <= 20:
            raise ValueError("The Tavily result limit must be between 1 and 20.")

        api_key = self._api_key or os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to the agent's .env file "
                "or set it in the environment before starting ADK."
            )

        request_body = json.dumps(
            {
                "query": query,
                "search_depth": "basic",
                "max_results": limit,
                "include_answer": False,
                "include_raw_content": False,
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=request_body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload: Any = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise RuntimeError(
                f"Tavily Search request failed with HTTP {error.code}."
            ) from error

        if not isinstance(payload, dict):
            raise RuntimeError("Tavily returned an invalid search response.")
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError(
                "Tavily search response did not contain a results list."
            )

        results: list[TavilySearchResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            url = item.get("url")
            content = item.get("content")
            if (
                not isinstance(title, str)
                or not isinstance(url, str)
                or not isinstance(content, str)
            ):
                continue
            if not url.startswith(("https://", "http://")):
                continue
            score = item.get("score")
            published_date = item.get("published_date")
            results.append(
                TavilySearchResult(
                    title=title,
                    url=url,
                    content=content,
                    score=score if isinstance(score, (int, float)) else None,
                    published_date=(
                        published_date
                        if isinstance(published_date, str)
                        else None
                    ),
                )
            )

        if not results:
            raise RuntimeError(
                f"Tavily returned no usable search results for {query!r}."
            )
        return results

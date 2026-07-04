"""Search without a search API: scrapes DuckDuckGo's HTML-only results page.

https://html.duckduckgo.com/html/ requires no API key and returns plain
HTML, which we parse directly, rather than calling a keyed search API.
"""

import os

import requests
from bs4 import BeautifulSoup

SEARCH_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = os.environ.get("SCRAPER_USER_AGENT", "agentic-platform-bot/0.1")


def parse_results(html: str, limit: int = 5) -> list[dict[str, str]]:
    """Pure parsing function, kept separate from the network call so it can
    be unit-tested against fixture HTML without hitting the network."""
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for link in soup.select("a.result__a")[:limit]:
        href = link.get("href", "")
        title = link.get_text(strip=True)
        if href and title:
            results.append({"title": title, "url": href})
    return results


def search(query: str, limit: int = 5) -> list[dict[str, str]]:
    resp = requests.post(
        SEARCH_URL,
        data={"q": query},
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    return parse_results(resp.text, limit=limit)

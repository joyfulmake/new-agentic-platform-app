"""Fetches a URL and extracts readable text via direct HTTP + HTML parsing
(no API involved) so agents get real page content rather than a mocked
summary."""

import os

import requests
from bs4 import BeautifulSoup

USER_AGENT = os.environ.get("SCRAPER_USER_AGENT", "agentic-platform-bot/0.1")
MAX_CHARS = 8000


def extract_text(html: str, max_chars: int = MAX_CHARS) -> str:
    """Pure parsing function, unit-testable against fixture HTML."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]


def fetch(url: str) -> str:
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=15,
    )
    resp.raise_for_status()
    return extract_text(resp.text)

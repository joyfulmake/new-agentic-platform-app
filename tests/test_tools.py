"""Tests the pure HTML-parsing logic of the scraping tools against fixture
HTML, so CI doesn't depend on internet access."""

from app.tools import web_fetch, web_search

DUCKDUCKGO_FIXTURE = """
<html><body>
  <div class="result">
    <a class="result__a" href="https://example.com/a">First Result</a>
  </div>
  <div class="result">
    <a class="result__a" href="https://example.com/b">Second Result</a>
  </div>
</body></html>
"""

PAGE_FIXTURE = """
<html>
<head><style>.x{color:red}</style></head>
<body>
  <nav>Site Nav</nav>
  <header>Header Junk</header>
  <main>
    <h1>Article Title</h1>
    <p>This is the real content that matters.</p>
  </main>
  <footer>Footer Junk</footer>
</body>
</html>
"""


def test_parse_results_extracts_titles_and_urls():
    results = web_search.parse_results(DUCKDUCKGO_FIXTURE)
    assert results == [
        {"title": "First Result", "url": "https://example.com/a"},
        {"title": "Second Result", "url": "https://example.com/b"},
    ]


def test_parse_results_respects_limit():
    results = web_search.parse_results(DUCKDUCKGO_FIXTURE, limit=1)
    assert len(results) == 1


def test_extract_text_strips_boilerplate():
    text = web_fetch.extract_text(PAGE_FIXTURE)
    assert "Article Title" in text
    assert "real content that matters" in text
    assert "Site Nav" not in text
    assert "Footer Junk" not in text

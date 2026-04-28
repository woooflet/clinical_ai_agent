
import requests
from typing import Optional

import config

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "ClinicalAIAgent/1.0"})

def register(mcp) -> None:
    """Attach all web search tools to the FastMCP instance."""

    @mcp.tool()
    def web_search(query: str, num_results: int = 5) -> list:
        """
        Search the web using the local SearXNG instance and return results.

        SearXNG must be running: docker run -d -p 8080:8080 --name searxng searxng/searxng

        Args:
            query: Search query string.
            num_results: Maximum number of results to return (default 5).

        Returns: list of {title, url, snippet, engine}
        """
        searxng_url = config.SEARXNG_URL.rstrip("/")
        num_results = min(int(num_results), 20)

        try:
            resp = _SESSION.get(
                f"{searxng_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "pageno": 1,
                    "language": "en",
                    "safesearch": 0,
                },
                timeout=10,
            )

            if resp.status_code == 200:
                data = resp.json()
                results = []
                for item in data.get("results", [])[:num_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "snippet": item.get("content", "")[:500],
                        "engine": item.get("engine", ""),
                    })
                return results if results else [{"status": "no_results", "query": query}]

            elif resp.status_code == 403:
                return [{
                    "error": "SearXNG returned 403   enable JSON format in settings.yml: "
                             "search: formats: [html, json]",
                    "query": query,
                }]
            else:
                return [{
                    "error": f"SearXNG returned HTTP {resp.status_code}",
                    "query": query,
                }]

        except requests.exceptions.ConnectionError:
            return [{
                "error": "SearXNG is not running. Start it with: "
                         "docker run -d -p 8080:8080 --name searxng searxng/searxng",
                "query": query,
                "searxng_url": searxng_url,
            }]
        except requests.exceptions.Timeout:
            return [{"error": "SearXNG timed out", "query": query}]
        except Exception as e:
            return [{"error": str(e), "query": query}]

    @mcp.tool()
    def scrape_url(url: str) -> dict:
        """
        Scrape a web page and return clean markdown using the local Firecrawl instance.

        Firecrawl must be running: clone mendableai/firecrawl, docker compose up -d
        (default port 3002). Falls back to a raw requests-based extraction if
        Firecrawl is unavailable.

        Args:
            url: URL of the page to scrape.

        Returns: {url, title, markdown_content, metadata, source}
        """
        firecrawl_url = config.FIRECRAWL_URL.rstrip("/")

        try:
            resp = _SESSION.post(
                f"{firecrawl_url}/v0/scrape",
                json={
                    "url": url,
                    "pageOptions": {
                        "onlyMainContent": True,
                        "includeHtml": False,
                    },
                },
                timeout=20,
            )

            if resp.status_code == 200:
                data = resp.json()
                page_data = data.get("data", {})
                return {
                    "url": url,
                    "title": page_data.get("metadata", {}).get("title", ""),
                    "markdown_content": (page_data.get("markdown") or "")[:8000],
                    "metadata": page_data.get("metadata", {}),
                    "source": "firecrawl",
                }
            else:
                raise requests.exceptions.ConnectionError(f"Firecrawl HTTP {resp.status_code}")

        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.Timeout:
            pass

        try:
            resp = _SESSION.get(url, timeout=15, allow_redirects=True)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "html" in content_type:
                text = _html_to_text(resp.text)
                return {
                    "url": url,
                    "title": _extract_html_title(resp.text),
                    "markdown_content": text[:8000],
                    "metadata": {"content_type": content_type},
                    "source": "fallback_requests",
                }
            else:
                return {
                    "url": url,
                    "title": "",
                    "markdown_content": resp.text[:8000],
                    "metadata": {"content_type": content_type},
                    "source": "fallback_requests",
                }

        except Exception as e:
            return {
                "url": url,
                "error": str(e),
                "note": "Firecrawl not running. Start with: "
                        "git clone https://github.com/mendableai/firecrawl && "
                        "cd firecrawl && docker compose up -d",
            }

def _extract_html_title(html: str) -> str:
    """Extract page title from raw HTML."""
    import re
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""

def _html_to_text(html: str) -> str:
    """Very lightweight HTML → plain text conversion without heavy dependencies."""
    import re
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<[^>]+>", " ", html)
    entities = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
                "&#39;": "'", "&nbsp;": " ", "&#160;": " "}
    for ent, char in entities.items():
        html = html.replace(ent, char)
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()

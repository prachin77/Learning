"""
Web Search & Scraping Tools for the GAIA Agent.
Uses DuckDuckGo (free, no API key) for search and BeautifulSoup for page scraping.
"""

import warnings
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_core.tools import tool

# Suppress harmless package warnings
warnings.filterwarnings("ignore")


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web and return the top results with titles, snippets, and URLs.
    Use this tool to find factual information, Wikipedia articles, or any web content.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return (default 5).

    Returns:
        A formatted string of search results with titles, URLs, and snippets.
    """
    try:
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results, safesearch="off"):
                    results.append(r)
        except Exception as ddg_err:
            print(f"DuckDuckGo search notice: {ddg_err}")

        # If DuckDuckGo returned no results, fallback to Wikipedia Search API
        if not results:
            try:
                import re
                wiki_url = "https://en.wikipedia.org/w/api.php"
                wiki_headers = {"User-Agent": "GAIA-Agent/1.0 (academic; evaluation; https://huggingface.co/spaces/Asynk/GAIA-Benchmark-System)"}
                wiki_params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": max_results,
                }
                w_resp = requests.get(wiki_url, params=wiki_params, headers=wiki_headers, timeout=10)
                if w_resp.status_code == 200:
                    w_items = w_resp.json().get("query", {}).get("search", [])
                    for item in w_items:
                        t = item.get("title", "")
                        snip = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                        page_url = f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}"
                        results.append({"title": t, "href": page_url, "body": snip})
            except Exception as w_err:
                print(f"Wikipedia search fallback notice: {w_err}")

        if not results:
            return f"No search results found for: {query}"

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('href', 'No URL')}\n"
                f"    Snippet: {r.get('body', 'No snippet')}"
            )
        return "\n\n".join(formatted)

    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def visit_webpage(url: str) -> str:
    """
    Visit a webpage and extract its text content.
    Use this after web_search to read the full content of a relevant page.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The extracted text content of the webpage (up to 30000 characters).
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Extract text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        # Limit output length (up to 30000 chars for complete information)
        if len(clean_text) > 30000:
            clean_text = clean_text[:30000] + "\n\n... [Content truncated at 30000 chars]"

        return clean_text

    except requests.exceptions.Timeout:
        return f"Error: Request timed out for URL: {url}"
    except requests.exceptions.RequestException as e:
        return f"Error visiting webpage {url}: {str(e)}"
    except Exception as e:
        return f"Unexpected error visiting {url}: {str(e)}"


@tool
def wikipedia_search(topic: str) -> str:
    """
    Search Wikipedia specifically for a topic and return the article content.
    Useful for factual questions about people, events, places, etc.

    Args:
        topic: The topic to search for on Wikipedia.

    Returns:
        The text content of the Wikipedia article.
    """
    try:
        # Use Wikipedia API for clean results with required User-Agent
        url = "https://en.wikipedia.org/w/api.php"
        headers = {
            "User-Agent": "GAIA-Agent/1.0 (https://huggingface.co/spaces/Asynk/GAIA-Benchmark-System; student@learning.org)"
        }
        params = {
            "action": "query",
            "format": "json",
            "titles": topic,
            "prop": "extracts",
            "explaintext": True,
            "redirects": 1,
        }
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                # Page not found, try search
                search_params = {
                    "action": "opensearch",
                    "format": "json",
                    "search": topic,
                    "limit": 1,
                }
                search_resp = requests.get(url, params=search_params, headers=headers, timeout=15)
                search_data = search_resp.json()
                if len(search_data) > 1 and search_data[1]:
                    # Retry with the first search result
                    params["titles"] = search_data[1][0]
                    response = requests.get(url, params=params, headers=headers, timeout=15)
                    data = response.json()
                    pages = data.get("query", {}).get("pages", {})
                    for pid, pg in pages.items():
                        extract = pg.get("extract", "")
                        if extract:
                            if len(extract) > 30000:
                                extract = extract[:30000] + "\n\n... [Truncated]"
                            return f"Wikipedia: {pg.get('title', topic)}\n\n{extract}"
                return f"No Wikipedia article found for: {topic}"

            extract = page.get("extract", "")
            if extract:
                if len(extract) > 30000:
                    extract = extract[:30000] + "\n\n... [Truncated]"
                return f"Wikipedia: {page.get('title', topic)}\n\n{extract}"

        return f"No content found in Wikipedia for: {topic}"

    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

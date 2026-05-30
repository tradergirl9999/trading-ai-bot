from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        lines = []
        for r in results:
            lines.append(f"**{r['title']}**\n{r['body']}\nSource: {r['href']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"


def news_search(query: str, max_results: int = 5) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
        if not results:
            return "No news found."
        lines = []
        for r in results:
            lines.append(f"**{r['title']}** ({r.get('date', 'N/A')})\n{r['body']}\nSource: {r['url']}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"News search failed: {e}"

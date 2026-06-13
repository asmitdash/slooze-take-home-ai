"""Challenge A — Web Search Agent.

Pipeline:
  1. User query
  2. DuckDuckGo search (free, no API key) -> top-N results
  3. Fetch and trim page text for each result
  4. Pass results + query to Claude (Bedrock) for synthesis
  5. Return answer + cited source URLs

Trade-offs:
  - DuckDuckGo (`ddgs`) chosen over Tavily/SerpAPI to avoid paid keys; the
    contract is identical so swapping in a higher-quality provider later is
    trivial.
  - Page-fetch is best-effort with a 4s timeout; we tolerate failures and
    fall back to the search snippet so the agent stays robust on flaky sites.
  - We deliberately keep the synthesis prompt strict about citing sources by
    URL to avoid hallucinated references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import httpx
from ddgs import DDGS

from .llm import BedrockClaude


SEARCH_RESULT_LIMIT = 5
PAGE_CHAR_BUDGET = 4000
TOTAL_CONTEXT_BUDGET = 12000


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    body: str = ""


SYSTEM_PROMPT = """You are a careful research assistant.
You will be given a user question and a list of web sources (title, URL, content excerpt).
Write a clear, factual answer grounded only in the provided sources. If the sources do not contain enough information, say so explicitly rather than inventing facts.

Output format:
Answer:
<2-6 sentence answer in plain prose>

Sources:
- <full URL 1>
- <full URL 2>

Cite only sources you actually used. Do not fabricate URLs."""


def _search(query: str, limit: int = SEARCH_RESULT_LIMIT) -> List[SearchResult]:
    out: List[SearchResult] = []
    with DDGS() as ddgs:
        for hit in ddgs.text(query, max_results=limit):
            out.append(
                SearchResult(
                    title=hit.get("title", ""),
                    url=hit.get("href") or hit.get("url", ""),
                    snippet=hit.get("body", ""),
                )
            )
    return out


def _strip_html(html: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _fetch_page(url: str, timeout: float = 4.0) -> str:
    try:
        r = httpx.get(
            url,
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (slooze-take-home/1.0)"},
        )
        if r.status_code != 200:
            return ""
        return _strip_html(r.text)[:PAGE_CHAR_BUDGET]
    except Exception:
        return ""


def _build_context(results: List[SearchResult]) -> str:
    chunks = []
    used = 0
    for i, r in enumerate(results, 1):
        body = r.body or r.snippet
        block = f"[{i}] {r.title}\nURL: {r.url}\nContent: {body}\n"
        if used + len(block) > TOTAL_CONTEXT_BUDGET:
            break
        chunks.append(block)
        used += len(block)
    return "\n".join(chunks)


def run(query: str, *, limit: int = SEARCH_RESULT_LIMIT) -> str:
    results = _search(query, limit=limit)
    if not results:
        return "Answer:\nNo search results were returned.\n\nSources:\n(none)"
    for r in results:
        r.body = _fetch_page(r.url)
    context = _build_context(results)
    user_prompt = f"User question: {query}\n\nWeb sources:\n{context}"
    llm = BedrockClaude()
    resp = llm.complete(system=SYSTEM_PROMPT, user=user_prompt, max_tokens=1200)
    return resp.text

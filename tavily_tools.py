"""Bound execution of official Tavily tools without handwritten HTTP requests."""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from time import sleep

from aiohttp import ClientError
from langchain_core.tools import ToolException, tool


class TavilySearchError(RuntimeError):
    """A sanitized search failure."""


def invoke_tavily(tool, arguments: dict, *, timeout: float = 30) -> dict:
    """Run the official async tool, with cancellation and at most one retry.

    A dedicated event loop supports callers that already have a running event loop.
    The integration hides Retry-After headers, so rate limits fail without retry.
    """
    async def invoke_once():
        return await asyncio.wait_for(tool.ainvoke(arguments), timeout=timeout)

    for attempt in (1, 2):
        print(f"[REQUEST] {tool.name}: attempt {attempt}/2; timeout={timeout:g}s", flush=True)
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = executor.submit(lambda: asyncio.run(invoke_once())).result()
            # langchain_tavily returns some failures inside an error dictionary.
            if isinstance(result, dict) and "error" in result:
                error = result["error"]
                raise error if isinstance(error, Exception) else ValueError(str(error))
            if not isinstance(result, dict) or not isinstance(result.get("results"), list):
                raise TavilySearchError("Tavily returned an invalid response.")
            return result
        except ToolException as exc:
            # Official tools signal empty results through ToolException.
            if str(exc).startswith("No search results found"):
                return {"results": []}
            raise TavilySearchError("Tavily tool execution failed.") from None
        except TavilySearchError:
            raise
        except Exception as exc:
            match = re.search(r"\bError (\d{3})\b", str(exc))
            status = int(match[1]) if match else None
            if status in (401, 403):
                raise TavilySearchError("Tavily rejected the credentials; check your API key.") from None
            if status in (432, 433):
                raise TavilySearchError("Tavily usage limit reached; check your account allowance.") from None
            if status == 429:
                raise TavilySearchError("Tavily rate limit reached; try again later.") from None
            transient = isinstance(exc, (ClientError, TimeoutError)) or (status is not None and 500 <= status < 600)
            if transient and attempt == 1:
                print("[RETRY] Temporary Tavily failure; retrying once in 1s.", flush=True)
                sleep(1)
                continue
            raise TavilySearchError(f"Tavily request failed after {attempt} attempt(s).") from None


def search_tools(store, owner, mode):
    role = 'discovery' if owner == 'discovery' else 'research'
    def search(query, topic):
        store.event(role, 'tool_call', 'search_news' if topic == 'news' else 'search_web', query)
        try:
            if not 3 <= len(query.strip()) <= 1000: raise ValueError('Query must be 3–1000 characters.')
            store.reserve(owner, topic)
        except ValueError as exc:
            return {'sources': [], 'error': str(exc), 'stop_searching': True}
        data = store.snapshot()
        if mode == 'offline':
            names = 'Atlas Suite, Beacon Office and Cedar Work' if owner == 'discovery' else next(c['name'] for c in data['candidates'] if c['id'] == owner)
            rows = [{'url': f'https://{owner}.example/{topic}', 'title': names,
                'content': names + ' offers shared documents, calendars, file storage and team messaging. Business plan: USD 20 per user per month. Targets small teams. Released calendar improvements.',
                'published_date': data['as_of'] if topic == 'news' else None}]
        else:
            from langchain_tavily import TavilySearch
            options = {'query': query}
            if topic == 'news':
                options.update(start_date=data['news_start'], end_date=(date.fromisoformat(data['as_of']) + timedelta(days=1)).isoformat())
            try:
                result = invoke_tavily(TavilySearch(max_results=5, topic=topic, search_depth='advanced',
                    include_answer=False, handle_tool_error=False), options)
                rows = result['results']
            except Exception as exc:
                store.event(role, 'search_error', topic, type(exc).__name__)
                return {'sources': [], 'error': 'Search unavailable: ' + type(exc).__name__}
        sources = []
        for row in rows:
            try: sources.append(store.add_source(owner, row, topic))
            except (ValueError, KeyError, TypeError): continue
        return {'sources': sources, 'searches_used': store.snapshot()['calls'].get(owner, 0)}

    @tool
    def search_web(query: str) -> dict:
        """Search Tavily for product, pricing, feature, positioning or competitor information. Choose a targeted query; refine it if necessary within budget."""
        return search(query, 'general')

    @tool
    def search_news(query: str) -> dict:
        """Search Tavily for recent news within the saved six-calendar-month window. Publication dates still require validation."""
        return search(query, 'news')

    return [search_web] if owner == 'discovery' else [search_web, search_news]

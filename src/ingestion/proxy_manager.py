"""
Free Proxy Manager using iplocate/free-proxy-list.
Fetches, caches, rotates, and validates working HTTP/HTTPS proxies to prevent IP rate limits
during academic literature and full-paper PDF retrieval.
"""

import time
import random
import logging
from pathlib import Path
from typing import List, Optional
import httpx
from src.config import settings

logger = logging.getLogger(__name__)

PROXY_HTTP_URL = "https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/http.txt"
PROXY_HTTPS_URL = "https://raw.githubusercontent.com/iplocate/free-proxy-list/main/protocols/https.txt"

class ProxyManager:
    """Manages pool of rotating free proxies from iplocate/free-proxy-list."""

    def __init__(self, cache_ttl_seconds: int = 1800):  # 30 min refresh
        self.cache_ttl = cache_ttl_seconds
        self.cache_file = settings.CACHE_DIR / "proxies.txt"
        self._proxies: List[str] = []
        self._last_refresh = 0.0

    async def get_proxies(self, force_refresh: bool = False) -> List[str]:
        """Load proxies from disk cache or fetch updated list from GitHub."""
        now = time.time()
        if not force_refresh and self._proxies and (now - self._last_refresh < self.cache_ttl):
            return self._proxies

        # Check disk cache
        if not force_refresh and self.cache_file.exists():
            file_mtime = self.cache_file.stat().st_mtime
            if now - file_mtime < self.cache_ttl:
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
                        if lines:
                            self._proxies = lines
                            self._last_refresh = file_mtime
                            logger.info(f"Loaded {len(self._proxies)} proxies from cache: {self.cache_file.name}")
                            return self._proxies
                except Exception as e:
                    logger.warning(f"Error reading proxy cache: {e}")

        # Fetch live from iplocate/free-proxy-list
        proxies = await self._fetch_proxies_from_github()
        if proxies:
            self._proxies = proxies
            self._last_refresh = now
            # Save to disk cache
            try:
                settings.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(proxies))
                logger.info(f"Saved {len(proxies)} proxies to cache: {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to cache proxies to file: {e}")

        return self._proxies

    async def _fetch_proxies_from_github(self) -> List[str]:
        """Download list of working proxies from iplocate/free-proxy-list repository."""
        logger.info("Fetching updated free proxies from iplocate/free-proxy-list...")
        all_proxies = set()

        async with httpx.AsyncClient(timeout=10.0) as client:
            for url in [PROXY_HTTP_URL, PROXY_HTTPS_URL]:
                try:
                    res = await client.get(url)
                    if res.status_code == 200:
                        lines = res.text.strip().splitlines()
                        for l in lines:
                            line = l.strip()
                            if line and ":" in line:
                                all_proxies.add(f"http://{line}")
                except Exception as e:
                    logger.warning(f"Failed to fetch proxy list from {url}: {e}")

        logger.info(f"Retrieved {len(all_proxies)} unique proxies from iplocate/free-proxy-list")
        return list(all_proxies)

    async def get_random_proxy(self) -> Optional[str]:
        """Return a random proxy from the available pool."""
        proxies = await self.get_proxies()
        if not proxies:
            return None
        return random.choice(proxies)

    async def get_working_client(self, timeout: float = 15.0) -> httpx.AsyncClient:
        """
        Produce an httpx AsyncClient configured with a working proxy.
        If proxies fail or are empty, gracefully returns client with direct connection.
        """
        proxy = await self.get_random_proxy()
        if proxy:
            try:
                return httpx.AsyncClient(proxy=proxy, timeout=timeout)
            except Exception as e:
                logger.warning(f"Could not configure client with proxy {proxy} ({e}). Using direct.")

        return httpx.AsyncClient(timeout=timeout)

# Global proxy manager instance
proxy_manager = ProxyManager()


from typing import Optional, Dict, Any
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def get_with_retry(
    session: aiohttp.ClientSession, url: str, headers: Optional[Dict[str, str]] = None
) -> Optional[str]:
    async with session.get(url, headers=headers) as resp:
        if resp.status != 200:
            raise aiohttp.ClientError(f"Status {resp.status} for {url}")
        return await resp.text()

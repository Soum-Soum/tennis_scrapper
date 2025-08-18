import asyncio
from typing import Optional, Dict

import aiohttp
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


def _check_response_status(status: int, url: str, exc_type):
    if status == 500 or status == 404:
        raise RuntimeError(f"Server error for {url} : Status code {status}")
    elif status != 200:
        raise exc_type(f"Status {status} for {url}")



@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=(
        retry_if_exception_type(aiohttp.ClientError) |
        retry_if_exception_type(asyncio.TimeoutError)
    ),
)
async def async_get_with_retry(
    session: aiohttp.ClientSession, url: str, headers: Optional[Dict[str, str]] = None
) -> Optional[str]:
    async with session.get(url, headers=headers) as resp:
        _check_response_status(resp.status, url, aiohttp.ClientError)
        return await resp.text()


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(requests.RequestException),
)
def sync_get_with_retry(
    url: str, headers: Optional[Dict[str, str]] = None
) -> Optional[str]:
    resp = requests.get(url, headers=headers)
    _check_response_status(resp.status_code, url, requests.RequestException)
    return resp.text

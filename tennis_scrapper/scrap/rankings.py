from tennis_scrapper.utils.http_utils import async_get_with_retry


import aiohttp
from bs4 import BeautifulSoup


async def scarp_dates(
    session: aiohttp.ClientSession, extension: str, year: int
) -> tuple[str, list[str]]:
    url = f"https://www.tennisexplorer.com/ranking/{extension}/{year}/"
    html = await async_get_with_retry(session, url, headers={"Accept": "text/html"})
    soup = BeautifulSoup(html, "lxml")
    date_select = soup.find("select", id="rform-date")
    options = date_select.find_all("option")
    options_values_date = [option["value"] for option in options if option["value"]]
    return url, options_values_date

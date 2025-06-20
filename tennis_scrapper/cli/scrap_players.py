import datetime
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

from conf.config import settings
from db.models import Gender, Player
from utils.http_utils import async_get_with_retry


def extract_player(html, player_detail_url_extension: str) -> Player:

    def find_div_by_text(divs, start_with: str):
        return next(
            filter(
                lambda d: d.get_text(strip=True).startswith(start_with),
                divs,
            ),
            None,
        )

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", "plDetail")
    td = table.find_all("td")[1]
    player_name = td.find("h3").get_text(strip=True)
    divs = td.find_all("div", class_="date")

    country_div = find_div_by_text(divs, "Country: ")
    country = (
        country_div.get_text(strip=True).replace("Country: ", "").strip()
        if country_div
        else "Unknown"
    )

    birth_date_div = find_div_by_text(divs, "Age: ")
    birth_date = (
        birth_date_div.get_text(strip=True).split("(")[1].replace(")", "").strip()
        if birth_date_div
        else "01.01.1970"
    )
    day, month, year = map(int, birth_date.split("."))
    birth_date = datetime.date(year=year, month=month, day=day)

    gender_div = find_div_by_text(divs, "Sex: ")
    gender = (
        Gender.from_string(gender_div.get_text(strip=True).replace("Sex: ", "").strip())
        if gender_div
        else "Unknown"
    )

    main_hand_div = find_div_by_text(divs, "Plays: ")
    main_hand = (
        main_hand_div.get_text(strip=True).replace("Plays: ", "").strip().upper()
        if main_hand_div
        else "UNKNOWN"
    )
    return Player(
        name=player_name,
        country=country,
        birth_date=birth_date,
        gender=gender,
        preferred_hand=main_hand,
        player_detail_url_extension=player_detail_url_extension,
    )


async def fetch_player(
    session: aiohttp.ClientSession, player_name: str, player_detail_url_extension: str
) -> Optional[Player]:
    url = f"{settings.base_url}/{player_detail_url_extension}"
    logger.info(f"Scraping player data from {player_name} : {url}")
    try:
        html: Optional[str] = await async_get_with_retry(
            session, url, headers={"Accept": "text/html"}
        )
    except RuntimeError as e:
        logger.error(f"Failed to fetch player data for {player_name}: {e}")
        return None
    player = extract_player(html, player_detail_url_extension)
    return player

import datetime
from typing import Optional
import aiohttp
from bs4 import BeautifulSoup
from loguru import logger
from sqlmodel import Session
from db.db_utils import engine
from conf.config import settings
from db.models import Gender, Player
from utils.str_utils import remove_digits


def extract_player(html) -> Player:

    def find_div_by_text(divs, start_with: str):
        return next(
            filter(
                lambda d: d.get_text(strip=True).startswith(start_with),
                divs,
            ),
            None,
        )

    soup = BeautifulSoup(html, "html.parser")
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
    height_div = find_div_by_text(divs, "Height ")
    height = (
        remove_digits(height_div.get_text(strip=True).split("/")[1])
        if height_div
        else "Unknown"
    )

    birth_date_div = find_div_by_text(divs, "Born: ")
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
        else "Unknown"
    )
    return Player(
        name=player_name,
        country=country,
        height=height,
        birth_date=birth_date,
        gender=gender,
        prefered_hand=main_hand,
    )


async def fetch_player(
    session: aiohttp.ClientSession, player_name: str, player_detail_url_extension: str
) -> Optional[Player]:
    url = f"{settings.base_url}/{player_detail_url_extension}"
    logger.info(f"Scraping player data from {player_name} : {url}")
    async with session.get(url, headers={"Accept": "text/html"}) as resp:
        if resp.status != 200:
            logger.error(f"Error while downloading {url}")
            return None
        html = await resp.text()
        player = extract_player(html)
        logger.success(f"Player data extracted for {player.name}")
        return player

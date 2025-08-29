from typing import Any, Coroutine
import aiohttp
from loguru import logger
from sqlmodel import Session
from tennis_scrapper.db.models import Gender, Player
from tennis_scrapper.scrap.utils import find_div_by_text


from bs4 import BeautifulSoup


import datetime

from tennis_scrapper.scrap.urls import get_player_detail_url
from tennis_scrapper.utils.http_utils import async_get_with_retry


def player_from_html(html: str, player_detail_url_extension: str) -> Player:

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
    if gender_div is not None:
        gender = Gender.from_string(
            gender_div.get_text(strip=True).replace("Sex: ", "").strip()
        )
    else:
        gender = "UNKNOWN"

    main_hand_div = find_div_by_text(divs, "Plays: ")
    if main_hand_div is not None:
        main_hand = (
            main_hand_div.get_text(strip=True).replace("Plays: ", "").strip().upper()
        )
    else:
        main_hand = "UNKNOWN"
    return Player(
        name=player_name,
        country=country,
        birth_date=birth_date,
        gender=gender,
        preferred_hand=main_hand,
        url_extension=player_detail_url_extension,
    )


async def scrap_player_details(
    db_session: Session, html_session: aiohttp.ClientSession, player_url_extension: str
) -> None:
    url = get_player_detail_url(player_url_extension)

    html = await async_get_with_retry(
        html_session, url, headers={"Accept": "text/html"}
    )

    if html is None:
        logger.error(f"Failed to fetch match data from {url}")
        return

    player = player_from_html(html, player_url_extension)

    db_session.add(player)
    db_session.commit()
    logger.success(f"Player {player.name} ({player.url_extension}) saved to DB.")


def get_players_scrapping_tasks(
    db_session: Session,
    html_session: aiohttp.ClientSession,
    all_unique_players_url_extentions: set,
) -> list[Coroutine[Any, Any, None]]:
    tasks = []
    for player_url_extension in all_unique_players_url_extentions:
        tasks.append(
            scrap_player_details(db_session, html_session, player_url_extension)
        )
    return tasks

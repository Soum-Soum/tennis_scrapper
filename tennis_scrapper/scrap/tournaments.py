import datetime

import aiohttp
from requests import Session
from db.models import Gender, Surface, Tournament


from bs4 import BeautifulSoup
from loguru import logger


from typing import Any, List, Coroutine

from scrap.urls import get_one_year_tournaments_url
from db.db_utils import insert_if_not_exists
from utils.http_utils import async_get_with_retry


def parse_surface(td: Any) -> Surface:
    span = td.find("span", title=True)
    if span is None:
        return Surface.UNKNOWN

    title = span["title"].strip().lower()
    if "indoor" in title:
        return Surface.INDOOR
    if "hard" in title:
        return Surface.HARD
    if "clay" in title:
        return Surface.CLAY
    if "grass" in title:
        return Surface.GRASS
    raise ValueError(f"Unexpected surface value : {title}")


def parse_date(date_str: str) -> datetime.date | None:
    try:
        day, month, year = date_str.split(".")
        year = int(year.replace("<br>", "").strip())
        month = int(month.strip())
        day = int(day.strip())
        return datetime.date(year, month, day)
    except ValueError:
        logger.error(f"Unable to parse date from string: {date_str}")
        return None


def parse_prize(prize_str: str) -> float:
    if not prize_str or prize_str.strip() == "-":
        return 0.0
    prize_str = prize_str.replace(",", "").replace("$", "").replace("€", "").strip()
    try:
        return float(prize_str)
    except ValueError:
        return 0.0


def tournaments_from_html(html: str, players_gender: Gender) -> List[Tournament]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="tournamentList")
    tournaments: List[Tournament] = []
    if not table:
        return tournaments
    rows_one = table.find_all("tr", class_="one")
    rows_two = table.find_all("tr", class_="two")
    rows_pairs = list(zip(rows_one[::2], rows_one[1::2])) + list(
        zip(rows_two[::2], rows_two[1::2])
    )
    i = 0

    for row1, row2 in rows_pairs:
        name = row1.find(class_="t-name").get_text(strip=True)
        date = parse_date(row1.find("td", class_="shortdate").get_text(strip=True))
        if date is None:
            logger.error(f"Unable to parse date for tournament: {name} in row {i + 1}")
            continue
        match_list_url_extension = row1.find(class_="t-name").find("a")["href"]

        surface = parse_surface(row1.find("td", class_="s-color"))
        prize = parse_prize(row1.find("td", class_="tr").get_text(strip=True))
        tournament = Tournament(
            name=name,
            year=date.year,
            surface=surface,
            cash_prize=prize,
            players_gender=players_gender,
            url_extension=match_list_url_extension,
        )
        tournaments.append(tournament)

    return tournaments


def get_default_tournament(year: int, players_gender: Gender) -> Tournament:
    return Tournament(
        name=f"Default {year} ({players_gender.name})",
        year=year,
        surface=Surface.UNKNOWN,
        cash_prize=0,
        players_gender=players_gender,
        url_extension="",
    )


async def scrap_tournaments(
    db_session: Session, html_session: aiohttp.ClientSession, gender: Gender, year: int
) -> None:
    url = get_one_year_tournaments_url(gender=gender, year=year)
    html = await async_get_with_retry(
        html_session, url, headers={"Accept": "text/html"}
    )
    tournaments = tournaments_from_html(html, gender)
    if not tournaments:
        logger.info(f"No tournaments found for {url}")
        return 0
    else:
        logger.success(f"Found {len(tournaments)} tournaments for {url}")
    tournaments.append(get_default_tournament(year, gender))

    insert_if_not_exists(db_session=db_session, table=Tournament, instances=tournaments)


def get_tournaments_scrapping_tasks(
    db_session: Session,
    html_session: aiohttp.ClientSession,
    from_date: datetime.date,
    to_date: datetime.date,
) -> List[Coroutine[Any, Any, None]]:
    years_diff = (to_date.year - from_date.year) + 1
    tasks = []
    for gender in Gender:
        for year in range(from_date.year, from_date.year + years_diff):
            tasks.append(scrap_tournaments(db_session, html_session, gender, year))
    return tasks

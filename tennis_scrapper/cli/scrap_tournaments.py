import asyncio
import datetime
from typing import List, Any

import aiohttp
import typer
from bs4 import BeautifulSoup
from loguru import logger

from conf.config import settings
from db.db_utils import insert_if_not_exists
from db.models import Gender, Tournament, Surface


def parse_prize(prize_str: str) -> float:
    if not prize_str or prize_str.strip() == "-":
        return 0.0
    prize_str = prize_str.replace(",", "").replace("$", "").replace("€", "").strip()
    try:
        return float(prize_str)
    except ValueError:
        return 0.0


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


def extract_tournaments(html: str, players_gender: Gender) -> List[Tournament]:
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


async def fetch_and_store_tournaments(
    session: aiohttp.ClientSession, url_year: str, year: int, players_gender: Gender
) -> int:
    logger.info(f"Scraping {url_year}")
    async with session.get(url_year) as resp:
        if resp.status != 200:
            logger.error(f"Error while downloading {url_year}")
            return 0
        html = await resp.text()
        tournaments = extract_tournaments(html, players_gender)
        if not tournaments:
            logger.info(f"No tournaments found for {url_year}")
            return 0
        tournaments.append(
            Tournament(
                name=f"Default {year} ({players_gender.name})",
                year=year,
                surface=Surface.UNKNOWN,
                cash_prize=0,
                players_gender=players_gender,
                url_extension="",
            )
        )

        insert_if_not_exists(Tournament, tournaments)

        logger.success(f"{len(tournaments)} tournaments inserted for {url_year}")
        return len(tournaments)


app = typer.Typer()


@app.command(
    help=(
        "Scrape tennis tournaments from the ATP and WTA websites for a given year range."
        " The scraped data is stored in the database. "
    )
)
def scrap_tournaments(
    from_year: int = typer.Option(1990, "--from", help="Start year (default: 1990)"),
    to_year: int = typer.Option(
        datetime.datetime.now().year,
        "--to",
        help="End year (default: current year)",
    ),
) -> None:
    tournament_calendar_base_url: str = f"{settings.base_url}/calendar"

    async def run_scraping() -> None:
        tasks: list = []
        async with aiohttp.ClientSession() as session:
            for extension in ["atp-men", "wta-women"]:
                players_gender: Gender = (
                    Gender.MEN if "atp" in extension else Gender.WOMAN
                )
                for year in range(from_year, to_year + 1):
                    url_year: str = (
                        f"{tournament_calendar_base_url}/{extension}/{year}/"
                    )
                    tasks.append(
                        fetch_and_store_tournaments(
                            session, url_year, year, players_gender
                        )
                    )
            await asyncio.gather(*tasks)

    asyncio.run(run_scraping())


if __name__ == "__main__":
    app()

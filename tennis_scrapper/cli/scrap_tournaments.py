import datetime
from typing import List, Any
from bs4 import BeautifulSoup
from sqlmodel import Session
from db.models import Gender, Tournament, Surface
from conf.config import settings
from loguru import logger
import typer
from db.db_utils import engine
import asyncio
import aiohttp


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
    except Exception:
        return None


def parse_surface(td: Any) -> Surface:
    span = td.find("span")
    if not span or not span.has_attr("title"):
        return Surface.HARD  # fallback
    title = span["title"].strip().lower()
    if "indoor" in title:
        return Surface.INDOOR
    if "hard" in title:
        return Surface.HARD
    if "clay" in title:
        return Surface.CLAY
    if "grass" in title:
        return Surface.GRASS
    return Surface.HARD


def extract_tournaments(html: str, players_gender: Gender) -> List[Tournament]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="tournamentList")
    tournaments: List[Tournament] = []
    if not table:
        return tournaments
    rows = table.find_all("tr")
    i = 0
    while i < len(rows):
        row = rows[i]
        if row.find("td", class_="shortdate"):
            # This is a tournament row
            i += 2  # skip doubles row
        else:
            i += 1
            continue

        tds = row.find_all(["td", "th"])
        if len(tds) < 6:
            continue
        date = parse_date(tds[0].get_text(strip=True))
        name = tds[1].get_text(strip=True)
        if name == "Tournament":
            continue  # Skip header row
        match_list_url_extension = tds[1].find("a")["href"]
        surface = parse_surface(tds[2])
        prize = parse_prize(tds[3].get_text(strip=True))
        if date is None:
            continue
        tournament = Tournament(
            name=name,
            date=date,
            surface=surface,
            cash_prize=prize,
            players_gender=players_gender,
            match_list_url_extension=match_list_url_extension,
        )
        tournaments.append(tournament)
    return tournaments


async def fetch_and_store_tournaments(
    session: aiohttp.ClientSession, url_year: str, year: int, players_gender: Gender
) -> int:
    logger.info(f"Scraping {url_year}")
    try:
        async with session.get(url_year) as resp:
            if resp.status != 200:
                logger.error(f"Error while downloading {url_year}")
                return 0
            html = await resp.text()
            tournaments = extract_tournaments(html, players_gender)
            if not tournaments:
                logger.info(f"No tournaments found for {url_year}")
                return 0
            with Session(engine) as db_session:
                db_session.add_all(tournaments)
                db_session.commit()
            logger.success(f"{len(tournaments)} tournaments inserted for {url_year}")
            return len(tournaments)
    except Exception as e:
        logger.error(f"Error for {url_year}: {e}")
        return 0


app = typer.Typer()


@app.command()
def scrap_tournaments(
    from_year: int = typer.Option(1990, "--from", help="Start year (default: 1990)"),
    to_year: int = typer.Option(
        datetime.datetime.now().year,
        "--to",
        help="End year (default: current year)",
    ),
) -> None:
    """
    Scrape tournaments from tennisexplorer URL for a range of years.
    """
    tournament_calendar_base_url: str = f"{settings.base_url}/calendar"

    async def run_scraping() -> None:
        tasks: list = []
        async with aiohttp.ClientSession() as session:
            for extention in ["atp-men", "wta-women"]:
                players_gender: Gender = (
                    Gender.MEN if "atp" in extention else Gender.WOMAN
                )
                for year in range(from_year, to_year + 1):
                    url_year: str = (
                        f"{tournament_calendar_base_url}/{extention}/{year}/"
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

import asyncio
import datetime
from typing import Optional, List

import aiohttp
import typer
from bs4 import BeautifulSoup
from loguru import logger
from sqlmodel import Session, func, select
from tqdm.asyncio import tqdm

from scrap.matches import MatchData, get_row_id_to_tournament_url_ext
from scrap.matches import get_row_pairs
from scrap.matches import pair_to_match
from scrap.players import player_from_html
from db.db_utils import engine, clear_table
from db.models import Gender, Match, Player
from tennis_scrapper.scrap.urls import get_match_list_page_url
from utils.http_utils import async_get_with_retry

app = typer.Typer()


async def get_player_from_url_extension(
    player_url_extension: str, client_session: aiohttp.ClientSession
) -> Player:
    logger.info(f"Scraping player data from {player_url_extension}")
    url = f"https://www.tennisexplorer.com/{player_url_extension}/"
    html = await async_get_with_retry(client_session, url)
    player = player_from_html(
        html=html, player_detail_url_extension=player_url_extension
    )
    return player


async def scrape_matches_from_url(
    date: datetime.date,
    url: str,
    session: aiohttp.ClientSession,
):
    with Session(engine) as db_session:
        html: Optional[str] = await async_get_with_retry(
            session, url, headers={"Accept": "text/html"}
        )

        matches = match_data_set_from_url(
            html=html, date=date, db_session=db_session, url=url
        )
        db_session.add_all(matches)
        db_session.commit()


def _get_date_interval(from_year: int, to_year: int):
    """Détermine la période à scraper, en tenant compte de la DB."""
    with Session(engine) as db_session:
        max_date_in_db = db_session.exec(select(func.max(Match.date))).one()
        logger.info(f"Max date in DB: {max_date_in_db}")

    extensions = ["atp-single", "wta-single"]

    start_date = datetime.date(from_year, 1, 1)
    if max_date_in_db:
        start_date = max(start_date, max_date_in_db + datetime.timedelta(days=1))
    end_date = min(datetime.date(to_year, 12, 31), datetime.date.today())

    return start_date, end_date, extensions


async def _scrape_matches(
    start_date: datetime.date, end_date: datetime.date, extensions: List[str]
):
    """Lance l'ensemble des tâches de scraping pour chaque date et extension."""
    semaphore = asyncio.Semaphore(10)
    url_template = "https://www.tennisexplorer.com/results/?type={extension}&year={year}&month={month}&day={day}"
    day_diff = (end_date - start_date).days + 1

    async def _scrape_day(date: datetime.date, extension: str):
        async with semaphore:
            url = url_template.format(
                extension=extension,
                year=date.year,
                month=date.month,
                day=date.day,
            )
            async with aiohttp.ClientSession() as session:
                try:
                    await scrape_matches_from_url(date=date, url=url, session=session)
                except Exception as e:
                    logger.error(f"Error scraping {date} ({extension}) : {url} : {e}")

    tasks = []
    for day_offset in range(day_diff):
        for extension in extensions:
            tasks.append(
                _scrape_day(start_date + datetime.timedelta(days=day_offset), extension)
            )

    await tqdm.gather(*tasks, desc="Scraping matches for each day", unit="day")


@app.command(
    help="Scrape all matches from Tennis Explorer for a given year range and store them in the database."
)
def scrap_all_matches(
    from_year: int = typer.Option(1990, "--from", help="Start year (default: 1990)"),
    to_year: int = typer.Option(datetime.datetime.now().year, "--to", help="End year"),
    clear_db: bool = typer.Option(
        False, "--clear-db", help="Clear the database before scraping"
    ),
):
    """Main command to scrape matches within a year range."""
    if clear_db:
        logger.info("Clearing the database...")
        clear_table(Match)

    start_date, end_date, extensions = _get_date_interval(from_year, to_year)
    logger.info(
        f"Scraping matches from {start_date} to {end_date} for extensions: {extensions}"
    )

    asyncio.run(
        _scrape_matches(start_date=start_date, end_date=end_date, extensions=extensions)
    )

import asyncio
from datetime import date, datetime, timedelta
from difflib import Match
import select
import aiohttp
from loguru import logger
import pandas as pd
from sqlmodel import Session, func
from tqdm.asyncio import tqdm
import typer

from db.db_utils import get_session, insert_if_not_exists
from db.models import Player, Tournament
from scrap.matches import MatchData, get_match_scrapping_tasks
from scrap.players import get_players_scrapping_tasks
from tennis_scrapper.scrap.tournaments import get_tournaments_scrapping_tasks


app = typer.Typer()


def get_new_players_url_extensions(
    db_session: Session, match_data_list: list[MatchData]
) -> set[str]:
    existing_players_url_extensions = db_session.exec(
        select(Player.url_extension).distinct()
    ).all()
    existing_players_url_extensions = set(existing_players_url_extensions)

    all_unique_players_url_extentions = set()
    for match_data in match_data_list:
        all_unique_players_url_extentions.add(match_data.player_1_url_extension)
        all_unique_players_url_extentions.add(match_data.player_2_url_extension)

    new_players_url_extensions = (
        all_unique_players_url_extentions - existing_players_url_extensions
    )
    return new_players_url_extensions


def get_intervals(
    db_sessoin: Session, from_date: date, to_date: date
) -> list[tuple[date, date]]:
    assert from_date <= to_date, "from_date doit être ≤ to_date"

    first_match_in_db_date, last_match_in_db_date = db_sessoin.exec(
        select(func.min(Match.date), func.max(Match.date))
    ).first()

    # Rien en base → tout est à couvrir
    if first_match_in_db_date is None and last_match_in_db_date is None:
        return [(from_date, to_date)]

    assert (
        first_match_in_db_date is not None and last_match_in_db_date is not None
    ), "Database should have at least one match"

    intervals: list[tuple[date, date]] = []

    # Intervalle à gauche (avant le premier match en DB)
    if from_date < first_match_in_db_date:
        left_end = min(to_date, first_match_in_db_date - timedelta(days=1))
        if from_date <= left_end:
            intervals.append((from_date, left_end))

    # Intervalle à droite (après le dernier match en DB)
    if to_date > last_match_in_db_date:
        right_start = max(from_date, last_match_in_db_date + timedelta(days=1))
        if right_start <= to_date:
            intervals.append((right_start, to_date))

    return intervals


async def process_one_interval(
    db_session: Session,
    http_session: aiohttp.ClientSession,
    from_date: date,
    to_date: date,
):
    logger.info(f"Start processing interval: {from_date} to {to_date}")

    tournaments = await tqdm.gather(
        *get_tournaments_scrapping_tasks(http_session, from_date, to_date),
        desc="Scraping tournaments data",
        unit="tournament",
    )
    tournaments = sum(tournaments, [])
    logger.info(
        f"Found {len(tournaments)} tournaments to merge with the existing ones in the database"
    )
    insert_if_not_exists(db_session=db_session, table=Tournament, instances=tournaments)

    match_data_list = await tqdm.gather(
        *get_match_scrapping_tasks(http_session, from_date, to_date),
        desc=f"Scraping matches data from {from_date} to {to_date}",
        unit="page",
    )

    match_data_list = sum(map(list, match_data_list), [])
    logger.info(f"Found {len(match_data_list)} new matches to store in the database")

    new_players_url_extensions = get_new_players_url_extensions(
        db_session, match_data_list
    )
    logger.info(
        f"Found {len(new_players_url_extensions)} new players to store in the database"
    )

    new_players = await tqdm.gather(
        *get_players_scrapping_tasks(http_session, new_players_url_extensions),
        desc="Scraping players data",
        unit="player",
    )
    logger.info(f"Found {len(new_players)} new players to store in the database")

    db_session.add_all(new_players)
    db_session.commit()
    logger.info(f"Inserted {len(new_players)} new players into the database")


async def scrap_all_data(from_date: date, to_date: date):
    with get_session() as db_session:
        async with aiohttp.ClientSession() as http_session:

            intervals = get_intervals(db_session, from_date, to_date)

            for from_date, to_date in intervals:

                await process_one_interval(db_session, http_session, from_date, to_date)


@app.command(help="Scrape all data from Tennis Explorer and store it in the database.")
def scrap_all(
    from_date: datetime = typer.Option(
        datetime(2010, 1, 1), help="The start date for the data scraping."
    ),
    to_date: datetime = typer.Option(
        datetime.today(), help="The end date for the data scraping."
    ),
):

    asyncio.run(scrap_all_data(from_date.date(), to_date.date()))


if __name__ == "__main__":
    app()

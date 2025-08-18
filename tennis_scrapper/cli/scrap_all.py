import asyncio
from datetime import date, datetime, timedelta
import aiohttp
from loguru import logger
from sqlmodel import Session, select, func
from tqdm.asyncio import tqdm
import typer

from db.db_utils import get_session
from db.models import Player, Match
from scrap.matches import get_match_scrapping_tasks
from scrap.tournaments import get_tournaments_scrapping_tasks
from scrap.players import get_players_scrapping_tasks


app = typer.Typer()


def get_new_players_url_extensions(
    db_session: Session,
) -> set[str]:
    players_in_db_url_extensions = db_session.exec(
        select(Player.url_extension).distinct()
    ).all()
    players_in_db_url_extensions = set(players_in_db_url_extensions)

    match_players_url_extensions = db_session.exec(
        select(Match.player_1_url_extension).union(
            select(Match.player_2_url_extension)
        )
    ).all()
    match_players_url_extensions = set(map(lambda x: x[0], match_players_url_extensions))

    return match_players_url_extensions - players_in_db_url_extensions


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

    await tqdm.gather(
        *get_tournaments_scrapping_tasks(db_session, http_session, from_date, to_date),
        desc="Scraping tournaments data",
        unit="tournament",
    )

    await tqdm.gather(
        *get_match_scrapping_tasks(db_session, http_session, from_date, to_date),
        desc=f"Scraping matches data from {from_date} to {to_date}",
        unit="page",
    )
    
    new_players_url_extensions = get_new_players_url_extensions(db_session)
    logger.info(
        f"Found {len(new_players_url_extensions)} new players to store in the database"
    )

    await tqdm.gather(
        *get_players_scrapping_tasks(db_session, http_session, new_players_url_extensions),
        desc="Scraping players data",
        unit="player",
    )


    
    
    # new_rankings = await tqdm.gather(
    #     *get_ranking_scrapping_tasks(http_session, from_date, to_date),
    #     desc="Scraping players rankings",
    #     unit="player",
    # )
    # logger.info(f"Found {len(new_rankings)} new rankings to store in the database")
    # db_session.add_all(new_rankings)
    # db_session.commit()


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

    try:
        asyncio.run(scrap_all_data(from_date.date(), to_date.date()))
    except Exception as e:
        logger.error(f"Error occurred while scraping data: {e}")


if __name__ == "__main__":
    app()

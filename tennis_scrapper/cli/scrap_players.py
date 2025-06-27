import asyncio
import datetime
from typing import Optional

import aiohttp
import typer
from bs4 import BeautifulSoup
from loguru import logger
from sqlmodel import Session, select
from tqdm.asyncio import tqdm

from conf.config import settings
from db.db_utils import engine, clear_table
from db.models import Gender, Player, Match
from utils.http_utils import async_get_with_retry


def extract_player(html: str, player_detail_url_extension: str) -> Player:

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
        url_extension=player_detail_url_extension,
    )


async def fetch_player(
    session: aiohttp.ClientSession,
    db_session: Session,
    player_detail_url_extension: str,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        url = f"{settings.base_url}/{player_detail_url_extension}"
        logger.info(f"Scraping player data from {player_detail_url_extension} : {url}")
        try:
            html: Optional[str] = await async_get_with_retry(
                session, url, headers={"Accept": "text/html"}
            )
        except RuntimeError as e:
            logger.error(
                f"Failed to fetch player data for {player_detail_url_extension}: {e}"
            )
            return
        player = extract_player(html, player_detail_url_extension)
        with db_session:
            db_session.add(player)
            db_session.commit()


app = typer.Typer()


@app.command()
def scrap_players(
    clear_db: bool = typer.Option(
        False,
        "--clear-db",
        help="Clear the database before scraping players",
    )
):

    with Session(engine) as db_session:
        if clear_db:
            logger.info("Clearing database before scraping players")
            clear_table(Player)

        all_unique_players_urls = (
            db_session.exec(select(Match.player_1_url_extension).distinct()).all()
            + db_session.exec(select(Match.player_2_url_extension).distinct()).all()
        )
        all_unique_players_urls = set(all_unique_players_urls)  # Remove duplicates
        already_scraped_players = db_session.exec(select(Player.url_extension)).all()
        all_unique_players_urls = all_unique_players_urls - set(already_scraped_players)
        logger.info(f"Found {len(all_unique_players_urls)} unique player URLs")

    async def launch_scraping():
        semaphore = asyncio.Semaphore(10)
        async with aiohttp.ClientSession() as session:
            with Session(engine) as db_session:
                tasks = []
                for player_url in all_unique_players_urls:
                    player_name = player_url.split("/")[0]
                    tasks.append(
                        fetch_player(session, db_session, player_url, semaphore)
                    )
                await tqdm.gather(*tasks, desc="Scraping players", unit="player")

    asyncio.run(launch_scraping())


if __name__ == "__main__":
    app()

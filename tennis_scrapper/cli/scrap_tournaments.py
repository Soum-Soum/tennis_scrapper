import asyncio
import datetime

import aiohttp
import typer
from loguru import logger

from conf.config import settings
from db.db_utils import insert_if_not_exists
from db.models import Gender, Tournament
from scrap.tournaments import scrap_tournaments_from_html, get_default_tournament


async def fetch_and_store_tournaments(
    session: aiohttp.ClientSession, url_year: str, year: int, players_gender: Gender
) -> int:
    logger.info(f"Scraping {url_year}")
    async with session.get(url_year) as resp:
        if resp.status != 200:
            logger.error(f"Error while downloading {url_year}")
            return 0
        html = await resp.text()
        tournaments = scrap_tournaments_from_html(html, players_gender)
        if not tournaments:
            logger.info(f"No tournaments found for {url_year}")
            return 0
        tournaments.append(get_default_tournament(year, players_gender))

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

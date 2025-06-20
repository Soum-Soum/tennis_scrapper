import asyncio
import datetime
from typing import Any, Coroutine

import aiohttp
import typer
from bs4 import BeautifulSoup
from loguru import logger
from sqlmodel import Session, select, func
from tqdm.asyncio import tqdm

from db.db_utils import engine, insert_if_not_exists
from db.models import Match, Ranking
from utils.http_utils import async_get_with_retry

app = typer.Typer()


async def scarp_dates(
    session: aiohttp.ClientSession, extension: str, year: int
) -> tuple[str, list[str]]:
    url = f"https://www.tennisexplorer.com/ranking/{extension}/{year}/"
    html = await async_get_with_retry(session, url, headers={"Accept": "text/html"})
    soup = BeautifulSoup(html, "lxml")
    date_select = soup.find("select", id="rform-date")
    options = date_select.find_all("option")
    options_values_date = [option["value"] for option in options if option["value"]]
    return url, options_values_date


async def scrape_one_date_ranking(
    session: aiohttp.ClientSession, url: str, date: str
) -> None:
    urls = [f"{url}/?date={date}&page={page_index}" for page_index in range(1, 50)]
    tasks = [scrap_one_date(session, url, date) for url in urls]
    all_rankings = await asyncio.gather(*tasks)
    all_rankings = sum(all_rankings, [])
    insert_if_not_exists(Ranking, all_rankings)


async def scrap_one_date(
    session: aiohttp.ClientSession, url: str, date: str
) -> list[Ranking]:
    # logger.info(f"Scraping ranking from {url}")
    html = await async_get_with_retry(session, url, headers={"Accept": "text/html"})
    soup = BeautifulSoup(html, "lxml")
    tbody = soup.find("tbody", class_="flags")
    if not tbody:
        return []
    trs = tbody.find_all("tr")
    rankings = []
    for tr in trs:
        rank = float(tr.find("td", class_="rank").get_text(strip=True))
        td_name = tr.find("td", class_="t-name")
        player_name = td_name.find("a").get_text(strip=True)
        player_url = td_name.find("a")["href"]
        points = float(tr.find("td", class_="long-point").get_text(strip=True))
        rankings.append(
            Ranking(
                date=datetime.datetime.strptime(date, "%Y-%m-%d").date(),
                rank=rank,
                player_name=player_name,
                player_url=player_url,
                points=points,
            )
        )

    return rankings


async def run_scrap_ranking(start_year: int, end_year: int) -> None:
    async with aiohttp.ClientSession() as session:
        tasks = []
        for year in range(start_year, end_year + 1):
            for extension in ["atp-men", "wta-women"]:
                tasks.append(
                    scarp_dates(
                        session,
                        extension=extension,
                        year=year,
                    )
                )
        dates = await tqdm.gather(*tasks, desc="Scraping dates for rankings")
        dates = dict(dates)

        tasks = []
        for url, dates in tqdm(
            dates.items(), desc="Scraping rankings for dates", total=len(dates)
        ):
            for date in tqdm(dates, desc=f"Scraping rankings for {url}", leave=False):
                await scrape_one_date_ranking(session, url, date)

        # await tqdm.gather(*tasks, desc="Scraping rankings for dates")


@app.command()
def scrap_ranking():
    with Session(engine) as session:
        min_date, max_date = session.exec(
            select(
                func.min(Match.date).label("date_min"),
                func.max(Match.date).label("date_max"),
            )
        ).first()

    start_year = min_date.year
    end_year = max_date.year
    asyncio.run(run_scrap_ranking(start_year, end_year))

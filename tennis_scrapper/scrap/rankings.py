from datetime import datetime, date
from typing import Any, Coroutine

from bs4 import BeautifulSoup
from loguru import logger
from sqlmodel import Session, select
from tqdm.asyncio import tqdm
from tennis_scrapper.db.models import Gender, Ranking
from tennis_scrapper.scrap.urls import get_ranking_date_url, get_ranking_url
from tennis_scrapper.db.db_utils import insert_if_not_exists
from tennis_scrapper.utils.http_utils import async_get_with_retry


import aiohttp


async def one_year_date_list(
    html_session: aiohttp.ClientSession, gender: Gender, year: int
) -> list[date]:
    url = get_ranking_date_url(gender, year)
    html = await async_get_with_retry(
        html_session, url, headers={"Accept": "text/html"}
    )
    soup = BeautifulSoup(html, "lxml")
    date_select = soup.find("select", id="rform-date")
    options = date_select.find_all("option")
    options_values_date = [option["value"] for option in options if option["value"]]
    options_values_date = list(
        map(lambda x: datetime.strptime(x, "%Y-%m-%d").date(), options_values_date)
    )
    return options_values_date


def get_dates_in_db(
    db_session: Session, gender: Gender, from_date: date, to_date: date
) -> list[date]:
    return db_session.exec(
        select(Ranking.date).where(
            Ranking.circuit == gender.circuit,
            Ranking.date >= from_date,
            Ranking.date <= to_date,
        )
    ).all()


async def get_dates_to_scrap(
    db_session: Session,
    session: aiohttp.ClientSession,
    from_date: date,
    to_date: date,
) -> dict[Gender, list[str]]:

    gender_to_dates = {}
    for gender in Gender:
        tasks = []
        for year in range(from_date.year, to_date.year + 1):
            tasks.append(one_year_date_list(session, gender, year))

        dates_lists = await tqdm.gather(
            *tasks,
            desc=f"Scraping rankings dates for {gender} {from_date.year}-{to_date.year}",
            unit="page",
        )
        scraped_dates = sum(dates_lists, [])
        logger.info(
            f"Found {len(scraped_dates)} rankings dates for {gender} {from_date.year}-{to_date.year}"
        )

        dates_in_db = get_dates_in_db(db_session, gender, from_date, to_date)

        dates_to_scrap = set(scraped_dates) - set(dates_in_db)

        gender_to_dates[gender] = list(dates_to_scrap)

    return gender_to_dates


def rankings_from_html(
    html: str,
    gender: Gender,
    date: date,
) -> list[Ranking]:
    soup = BeautifulSoup(html, "lxml")
    tbody = soup.find("tbody", class_="flags")
    if not tbody:
        return []
    trs = tbody.find_all("tr")
    rankings = []

    for tr in trs:
        rank = int(float(tr.find("td", class_="rank").get_text(strip=True)))
        td_name = tr.find("td", class_="t-name")
        player_name = td_name.find("a").get_text(strip=True)
        player_detail_url_extension = td_name.find("a")["href"]
        points = int(float(tr.find("td", class_="long-point").get_text(strip=True)))
        rankings.append(
            Ranking(
                date=date,
                rank=rank,
                player_name=player_name,
                player_detail_url_extension=player_detail_url_extension,
                points=points,
                circuit=gender.circuit,
            )
        )

    return rankings


async def scrape_rankings_one_page(
    db_session: Session,
    html_session: aiohttp.ClientSession,
    gender: Gender,
    date: date,
    page_index: int,
) -> None:
    url = get_ranking_url(gender, date, page_index=page_index)
    html = await async_get_with_retry(
        html_session, url, headers={"Accept": "text/html"}
    )
    all_rankings = rankings_from_html(html, gender, date)

    insert_if_not_exists(db_session, table=Ranking, instances=all_rankings)


def get_ranking_scrapping_tasks(
    db_session: Session,
    session: aiohttp.ClientSession,
    gender_to_dates: dict[Gender, list[date]],
) -> list[Coroutine[Any, Any, None]]:

    tasks = []
    for gender, dates in gender_to_dates.items():
        for current_date in dates:
            tasks.extend(
                [
                    scrape_rankings_one_page(
                        db_session, session, gender, current_date, page_index
                    )
                    for page_index in range(1, 50)
                ]
            )

    return tasks

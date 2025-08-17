from datetime import datetime, date
from typing import Any, Coroutine

from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm
from db.models import Gender, Ranking
from scrap.urls import get_ranking_date_url, get_ranking_url
from utils.http_utils import async_get_with_retry


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


async def scrap_dates(
    session: aiohttp.ClientSession, from_date: date, to_date: date
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
            dates = sum(dates_lists, [])
            gender_to_dates[gender] = dates

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
        rank = float(tr.find("td", class_="rank").get_text(strip=True))
        td_name = tr.find("td", class_="t-name")
        player_name = td_name.find("a").get_text(strip=True)
        player_detail_url_extension = td_name.find("a")["href"]
        points = float(tr.find("td", class_="long-point").get_text(strip=True))
        rankings.append(
            Ranking(
                date=date(),
                rank=rank,
                player_name=player_name,
                player_detail_url_extension=player_detail_url_extension,
                points=points,
                circuit=gender.circuit,
            )
        )

    return rankings


async def scrape_rankings_one_page(
    session: aiohttp.ClientSession, gender: Gender, date: date, page_index: int
) -> list[Ranking]:
    url = get_ranking_url(gender, date, page_index=page_index)
    html = await async_get_with_retry(session, url, headers={"Accept": "text/html"})
    all_rankings = rankings_from_html(html, gender, date)
    return all_rankings


def get_ranking_scrapping_tasks(
    session: aiohttp.ClientSession, from_date: date, to_date: date
) -> list[Coroutine[Any, Any, list[Ranking]]]:
    gender_to_dates = scrap_dates(session, from_date, to_date)

    tasks = []
    for gender, dates in gender_to_dates.items():
        for date in dates:
            tasks.extend(
                [
                    scrape_rankings_one_page(session, gender, date, page_index)
                    for page_index in range(1, 50)
                ]
            )

    return tasks

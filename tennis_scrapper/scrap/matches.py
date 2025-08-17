import datetime
import aiohttp
from bs4 import BeautifulSoup
from loguru import logger


from typing import Any, Coroutine, List, Optional

from pydantic import BaseModel

from db.models import Gender


from tennis_scrapper.scrap.urls import get_match_list_page_url
from tennis_scrapper.utils.http_utils import async_get_with_retry


class MatchData(BaseModel):
    date: datetime.date
    tournament_url_extension: str
    player_1_url_extension: str
    player_2_url_extension: str
    score: str
    player_1_odds: float
    player_2_odds: float


def parse_score(tr1, tr2) -> str:
    def remove_sup(s: str) -> str:
        if "<sup>" not in s:
            return s

        return s[: s.index("<sup>")].strip()

    def extract_score(tr) -> List[str]:
        tds = tr.find_all("td", class_="score")
        scores = list(
            filter(
                lambda s: len(s.strip()) > 0,
                map(lambda s: remove_sup(s.decode_contents()), tds),
            )
        )
        return scores

    scores1 = extract_score(tr1)
    scores2 = extract_score(tr2)

    if len(scores1) != len(scores2):
        logger.warning(
            f"Mismatch in score lengths: {scores1} vs {scores2}, {tr1}, {tr2}"
        )
        min_len = min(len(scores1), len(scores2))
        scores1 = scores1[:min_len]
        scores2 = scores2[:min_len]

    scores_str = " ".join(f"{s1}-{s2}" for s1, s2 in zip(scores1, scores2))
    return scores_str.strip()


# def get_default_tournament(
#     year: int,
#     players_gender: Gender,
#     db_session: Session,
# ) -> Tournament:
#     return db_session.exec(
#         select(Tournament).where(
#             col(Tournament.name).contains("Default"),
#             Tournament.year == year,
#             Tournament.players_gender == players_gender,
#         )
#     ).first()


# def get_tournament(
#     url_extension: Optional[str],
#     year: int,
#     players_gender: Gender,
#     db_session: Session,
# ) -> Tournament:
#     tournament = db_session.exec(
#         select(Tournament).where(
#             Tournament.url_extension == url_extension,
#             Tournament.year == year,
#             Tournament.players_gender == players_gender,
#         )
#     ).first()
#     if tournament is not None:
#         return tournament
#     else:
#         return get_default_tournament(year, players_gender, db_session)


def get_row_id_to_tournament_url_ext(
    trs: list, players_gender: Gender, date: datetime.date
) -> dict[int, str]:
    heads = list(
        filter(
            lambda x: x[1].has_attr("class") and "head" in x[1]["class"],
            enumerate(trs),
        )
    )

    row_id_to_tournament = {}
    for row_id, head_row in heads:
        tournament_url_extension_a = head_row.find("td", class_="t-name").find("a")

        if not tournament_url_extension_a:
            tournament_url_extension = f"DEFAULT-{players_gender}-{date.year}"
        else:
            tournament_url_extension = tournament_url_extension_a["href"]

        row_id_to_tournament[row_id] = tournament_url_extension

    return row_id_to_tournament


def get_row_pairs(trs: list) -> List[tuple]:
    row_one = list(
        filter(
            lambda tr: tr.has_attr("id")
            and tr["id"].startswith("r")
            and not tr["id"].endswith("b"),
            trs,
        )
    )
    row_two_and_id = list(
        filter(
            lambda x: x[1].has_attr("id")
            and x[1]["id"].startswith("r")
            and x[1]["id"].endswith("b"),
            enumerate(trs),
        )
    )

    row_pairs = list(zip(row_one, row_two_and_id))

    return row_pairs


def scrap_odds(row1) -> tuple[Optional[float], Optional[float]]:
    res = row1.find_all("td", class_="course")
    if len(res) == 2:
        player_1_odds_str = res[0].get_text(strip=True).strip()
        player_2_odds_str = res[1].get_text(strip=True).strip()
    elif len(res) == 1:
        player_1_odds_str = (
            row1.find("td", class_="coursew").get_text(strip=True).strip()
        )
        player_2_odds_str = (
            row1.find("td", class_="course").get_text(strip=True).strip()
        )

    else:
        raise ValueError(f"Unexpected number of odds columns: {len(res)}")

    player_1_odds = float(player_1_odds_str) if player_1_odds_str else None
    player_2_odds = float(player_2_odds_str) if player_2_odds_str else None

    return player_1_odds, player_2_odds


def pair_to_match(
    tournament_url_extension: str,
    row1: BeautifulSoup,
    row2: BeautifulSoup,
    date: datetime.date,
) -> Optional[MatchData]:

    try:
        player_1_url_extension = row1.find("td", class_="t-name").find("a")["href"]
        player_2_url_extension = row2.find("td", class_="t-name").find("a")["href"]
    except TypeError:
        logger.warning(f"Unable to find player URLs -> skipping row")
        return None

    player_1_odds, player_2_odds = scrap_odds(row1)

    score = parse_score(row1, row2)

    return MatchData(
        date=date,
        tournament_url_extension=tournament_url_extension,
        player_1_url_extension=player_1_url_extension,
        player_2_url_extension=player_2_url_extension,
        score=score,
        player_1_odds=player_1_odds,
        player_2_odds=player_2_odds,
    )


def match_data_set_from_html(
    html: str, date: datetime.date, gender: Gender
) -> set[MatchData]:
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", class_="result")

    trs = table.find_all("tr")

    row_id_to_tournament = get_row_id_to_tournament_url_ext(
        trs=trs, gender=gender, date=date
    )

    row_pairs = get_row_pairs(trs)

    matches = set()

    for row1, (row_two_id, row2) in row_pairs:
        tournament = row_id_to_tournament[
            max(filter(lambda x: x <= row_two_id, row_id_to_tournament.keys()))
        ]
        match_data = pair_to_match(
            tournament=tournament,
            row1=row1,
            row2=row2,
            date=date,
        )
        if match_data:
            matches.add(match_data)

    return matches


async def scrap_match_data(
    html_session: aiohttp.ClientSession,
    gender: Gender,
    date: datetime.date,
) -> set[MatchData]:
    url = get_match_list_page_url(gender, date)
    html = await async_get_with_retry(
        html_session, url, headers={"Accept": "text/html"}
    )
    return match_data_set_from_html(html, date, gender)


def get_match_scrapping_tasks(
    html_session: aiohttp.ClientSession,
    from_date: datetime.date,
    to_date: datetime.date,
) -> list[Coroutine[Any, Any, set[MatchData]]]:
    tasks = []
    day_diff = (to_date - from_date).days
    for day in range(day_diff + 1):
        for gender in Gender:
            date = from_date + datetime.timedelta(days=day)
            tasks.append(scrap_match_data(html_session, gender, date))

    return tasks

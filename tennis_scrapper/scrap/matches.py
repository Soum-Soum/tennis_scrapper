from collections import defaultdict
import datetime
import aiohttp
from bs4 import BeautifulSoup
from loguru import logger


from typing import Any, Coroutine, List, Optional

from sqlmodel import Session

from db.models import Gender, Match


from scrap.urls import get_match_result_list_page_url
from utils.http_utils import async_get_with_retry


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


def get_row_id_to_tournament_url_ext(
    trs: list, gender: Gender, date: datetime.date
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
            tournament_url_extension = f"DEFAULT-{gender}-{date.year}"
        else:
            tournament_url_extension = tournament_url_extension_a["href"]

        row_id_to_tournament[row_id] = tournament_url_extension

    return row_id_to_tournament


def get_row_pairs(trs: list) -> List[tuple]:
    all_pairs = []

    for letter in ["r", "s"]:
        row_one = list(
            filter(
                lambda tr: tr.has_attr("id")
                and tr["id"].startswith(letter)
                and not tr["id"].endswith("b"),
                trs,
            )
        )
        row_two_and_id = list(
            filter(
                lambda x: x[1].has_attr("id")
                and x[1]["id"].startswith(letter)
                and x[1]["id"].endswith("b"),
                enumerate(trs),
            )
        )

        row_pairs = list(zip(row_one, row_two_and_id))
        all_pairs.extend(row_pairs)

    return all_pairs


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
    gender: Gender,
) -> Optional[Match]:

    try:
        player_1_url_extension = row1.find("td", class_="t-name").find("a")["href"]
        player_2_url_extension = row2.find("td", class_="t-name").find("a")["href"]
    except TypeError:
        logger.warning("Unable to find player URLs -> skipping row")
        return None

    player_1_odds, player_2_odds = scrap_odds(row1)

    score = parse_score(row1, row2)

    return Match(
        date=date,
        tournament_url_extension=tournament_url_extension,
        player_1_url_extension=player_1_url_extension,
        player_2_url_extension=player_2_url_extension,
        score=score,
        player_1_odds=player_1_odds,
        player_2_odds=player_2_odds,
        players_gender=gender,
    )


def extract_matches_from_table(
    table: BeautifulSoup, date: datetime.date, gender: Gender
) -> set[Match]:
    trs = table.find_all("tr")

    row_id_to_tournament_url = get_row_id_to_tournament_url_ext(
        trs=trs, gender=gender, date=date
    )

    row_pairs = get_row_pairs(trs)

    matches = set()

    for row1, (row_two_id, row2) in row_pairs:
        tournament_url_extension = row_id_to_tournament_url[
            max(filter(lambda x: x <= row_two_id, row_id_to_tournament_url.keys()))
        ]
        match_data = pair_to_match(
            tournament_url_extension=tournament_url_extension,
            row1=row1,
            row2=row2,
            date=date,
            gender=gender,
        )
        if match_data:
            matches.add(match_data)

    return matches


def match_data_set_from_html(
    html: str, date: datetime.date, gender: Gender
) -> set[Match]:
    soup = BeautifulSoup(html, "lxml")

    table = soup.find("table", class_="result")

    return extract_matches_from_table(table, date, gender)


async def scrap_match_data(
    db_session: Session,
    html_session: aiohttp.ClientSession,
    gender: Gender,
    date: datetime.date,
) -> None:
    url = get_match_result_list_page_url(gender, date)
    html = await async_get_with_retry(
        html_session, url, headers={"Accept": "text/html"}
    )
    if html is None:
        logger.error(f"Failed to fetch match data from {url}")
        return

    matches = match_data_set_from_html(html, date, gender)

    if len(set([m.match_id for m in matches])) != len(matches):
        logger.error(f"Duplicate match IDs found @ {url}")
        id_to_count = defaultdict(int)
        for m in matches:
            id_to_count[m.match_id] += 1

        matches = list(filter(lambda m: id_to_count[m.match_id] == 1, matches))
        if not matches:
            return

    db_session.add_all(matches)
    db_session.commit()
    logger.success(
        f"{len(matches)} matches inserted in DB for {gender.player_url_extension} on {date}"
    )


def get_match_scrapping_tasks(
    db_session: Session,
    html_session: aiohttp.ClientSession,
    from_date: datetime.date,
    to_date: datetime.date,
) -> list[Coroutine[Any, Any, None]]:
    tasks = []
    day_diff = (to_date - from_date).days
    for day in range(day_diff + 1):
        for gender in Gender:
            date = from_date + datetime.timedelta(days=day)
            tasks.append(scrap_match_data(db_session, html_session, gender, date))

    return tasks

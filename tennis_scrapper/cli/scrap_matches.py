import datetime
from typing import List, Optional
from bs4 import BeautifulSoup
from sqlmodel import Session
from tqdm import tqdm
from db.models import Match, Player, Tournament
from conf.config import settings
from loguru import logger
import typer
from db.db_utils import get_table, engine
import asyncio
import aiohttp

from cli.scrap_players import fetch_player
from utils.str_utils import remove_digits, remove_punctuation


def parse_players_names(tr1, tr2) -> tuple[tuple[str, str], tuple[str, str]]:
    def extrac_name(tr):
        td = tr.find("td", "t-name")
        raw_name = td.get_text(strip=True)
        clean_name = remove_punctuation(remove_digits(raw_name)).strip()
        a = td.find("a")
        if not hasattr(a, "href"):
            return clean_name, None
        player_url_extention = td.find("a")["href"].strip()
        return clean_name, player_url_extention

    return extrac_name(tr1), extrac_name(tr2)


def parse_date(tr1, tournament: Tournament) -> datetime.datetime:
    td = tr1.find("td", ["first", "time"])
    day, month, houre = td.get_text(strip=True).split(".")
    houre, minute = houre.split(":")
    houre = houre if houre.isdigit() else "00"
    minute = minute if minute.isdigit() else "00"
    year = tournament.date.year
    return datetime.datetime(
        year=int(year),
        month=int(month),
        day=int(day),
        hour=int(houre),
        minute=int(minute),
    )


def parse_round(tr1) -> str:
    td = tr1.find("td", title=True)
    if not td:
        return "Unknown"
    return td["title"].strip()


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
                map(lambda s: s.get_text(strip=True), tds),
            )
        )
        scores = list(map(remove_sup, scores))
        return scores

    scores1 = extract_score(tr1)
    scores2 = extract_score(tr2)
    assert len(scores1) == len(
        scores2
    ), f"Mismatch in score lengths: {scores1} vs {scores2}, {tr1}, {tr2}"
    scores_str = " ".join(f"{s1}-{s2}" for s1, s2 in zip(scores1, scores2))
    return scores_str.strip()


def extract_odds(tr1) -> tuple[float, float]:
    tds = tr1.find_all("td", class_="course")
    player_1_odd = tds[0].get_text(strip=True).replace(",", ".")
    player_2_odd = tds[1].get_text(strip=True).replace(",", ".")
    player_1_odd = float(player_1_odd) if player_1_odd.isdigit() else 0.0
    player_2_odd = float(player_2_odd) if player_2_odd.isdigit() else 0.0
    return player_1_odd, player_2_odd


async def player_from_url(
    player_name: str,
    player_url: str,
    players: dict[str, Player],
    session: aiohttp.ClientSession,
) -> Optional[Player]:
    player = players.get(player_url)
    if player is None:
        player = await fetch_player(
            session=session,
            player_name=player_name,
            player_detail_url_extension=player_url,
        )
    return player


async def extract_matches_and_players(
    html: str,
    session: aiohttp.ClientSession,
    tournament: Tournament,
    players: dict[str, Player],
) -> tuple[tuple[list[Match], Player, Player], dict[str, Player]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", class_="result")
    matches: List[Match] = []
    if not table:
        return matches
    rows = table.find_all("tr")

    row_one_list = list(
        filter(
            lambda r: r.has_attr("id") and not r["id"].endswith("b"),
            rows,
        )
    )

    row_two_list = list(
        filter(
            lambda r: r.has_attr("id") and r["id"].endswith("b"),
            rows,
        )
    )

    assert len(row_one_list) == len(row_two_list), "Mismatch in row counts"
    row_pairs = zip(row_one_list, row_two_list)

    match_and_players = []
    for row1, row2 in row_pairs:
        (player1_name, player1_url), (player2_name, player2_url) = parse_players_names(
            row1, row2
        )

        player_1 = await player_from_url(
            player_name=player1_name,
            player_url=player1_url,
            players=players,
            session=session,
        )
        player_2 = await player_from_url(
            player_name=player2_name,
            player_url=player2_url,
            players=players,
            session=session,
        )
        if player_1 is None or player_2 is None:
            logger.warning(
                f"Player not found: {player1_name} ({player1_url}) or {player2_name} ({player2_url})"
            )
            continue
        players[player1_url] = player_1
        players[player2_url] = player_2

        match_date = parse_date(row1, tournament)
        match_round = parse_round(row1)
        match_score = parse_score(row1, row2)
        player_1_odds, player_2_odds = extract_odds(row1)
        match = Match(
            tournament_id=tournament.tournament_id,
            date=match_date,
            player1_id=player_1.player_id,
            player2_id=player_2.player_id,
            score=match_score,
            player_1_odds=player_1_odds,
            player_2_odds=player_2_odds,
            round=match_round,
        )
        match_and_players.append((match, player_1, player_2))

    return match_and_players, players


app = typer.Typer()


@app.command()
def scrap_matches() -> None:
    tournaments = get_table(Tournament)

    async def run_scraping():
        async with aiohttp.ClientSession() as session:
            players = {}
            for tournament in tqdm(
                tournaments,
                desc="Scraping matches",
                unit="tournament",
            ):
                url = f"{settings.base_url}/{tournament.match_list_url_extension}"
                logger.info(f"Scraping matches from {tournament.name} : {url}")
                async with session.get(url, headers={"Accept": "text/html"}) as resp:
                    if resp.status != 200:
                        logger.error(f"Error while downloading {url}")
                        continue
                    html = await resp.text()
                    match_and_players, players = await extract_matches_and_players(
                        html=html,
                        session=session,
                        tournament=tournament,
                        players=players,
                    )
                    if not match_and_players:
                        logger.info(f"No matches found for {url}")
                        continue
                    with Session(engine) as db_session:
                        for match, player1, player2 in match_and_players:
                            db_session.merge(player1)
                            db_session.merge(player2)
                            db_session.add(match)

                        db_session.commit()
                    logger.success(
                        f"{len(match_and_players)} matches inserted for {url}"
                    )

    asyncio.run(run_scraping())


if __name__ == "__main__":
    app()

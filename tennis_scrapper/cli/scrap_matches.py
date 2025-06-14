import datetime
from typing import List, Optional, Any, Coroutine
from bs4 import BeautifulSoup
from sqlmodel import Session
from tqdm import tqdm
from db.models import Match, Player, Tournament
from conf.config import settings
from loguru import logger
import typer
from db.db_utils import get_table, engine, insert_if_not_exists
import asyncio
import aiohttp

from cli.scrap_players import fetch_player
from utils.http_utils import get_with_retry
from utils.str_utils import remove_digits, remove_punctuation


def parse_players_names(tr1, tr2) -> tuple[tuple[str, str], tuple[str, str]]:
    def extrac_name(tr):
        td = tr.find("td", "t-name")
        raw_name = td.get_text(strip=True)
        clean_name = remove_punctuation(remove_digits(raw_name)).strip()
        a = td.find("a")
        if not hasattr(a, "href"):
            return clean_name, None
        player_url_extension = td.find("a")["href"].strip()
        return clean_name, player_url_extension

    return extrac_name(tr1), extrac_name(tr2)


def parse_date(tr1, tournament: Tournament) -> datetime.datetime:
    td = tr1.find("td", ["first", "time"])
    day, month, hour = td.get_text(strip=True).split(".")
    hour, minute = hour.split(":")
    hour = hour if hour.isdigit() else "00"
    minute = minute if minute.isdigit() else "00"
    year = tournament.date.year
    return datetime.datetime(
        year=int(year),
        month=int(month),
        day=int(day),
        hour=int(hour),
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
    players: dict[str, Optional[Player]],
    session: aiohttp.ClientSession,
) -> Optional[Player]:
    if player_url in players:
        return players[player_url]

    player = await fetch_player(
        session=session,
        player_name=player_name,
        player_detail_url_extension=player_url,
    )

    players[player_url] = player
    return player


async def extract_matches_and_players(
    html: str,
    session: aiohttp.ClientSession,
    tournament: Tournament,
    players: dict[str, Player],
) -> tuple[list[tuple[Match, Player, Player]], dict[str, Player]]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="result")
    rows = table.find_all("tr")

    row_one_list = [r for r in rows if r.has_attr("id") and not r["id"].endswith("b")]
    row_two_list = [r for r in rows if r.has_attr("id") and r["id"].endswith("b")]

    assert len(row_one_list) == len(row_two_list), "Mismatch in row counts"
    row_pairs = list(zip(row_one_list, row_two_list))

    match_and_players: list[tuple[Match, Player, Player]] = []

    async def process_row_pair(row1, row2):
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
            return None

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
        return match, player_1, player_2

    tasks = [process_row_pair(r1, r2) for r1, r2 in row_pairs]
    results = await asyncio.gather(*tasks)

    match_and_players = [res for res in results if res is not None]

    return match_and_players, players


app = typer.Typer()


async def run_scraping(tournaments: list[Tournament]) -> None:
    async with aiohttp.ClientSession() as session:
        players: dict[Any, Any] = {}
        for tournament in tqdm(
            tournaments,
            desc="Scraping tournament matches",
            unit="tournament",
        ):
            base_url = f"{settings.base_url}/{tournament.match_list_url_extension}"
            extensions = ["/?phase=main", "/?phase=qualification"]

            all_match_and_players = []
            for extension in extensions:
                url = f"{base_url}{extension}"  # <-- corrige la concaténation ici
                logger.info(f"Scraping matches from {tournament.name} : {url}")

                html: Optional[str] = await get_with_retry(
                    session, url, headers={"Accept": "text/html"}
                )

                if html is None:
                    logger.error(f"Failed to download {url} after retries")
                    continue

                match_and_players, players = await extract_matches_and_players(
                    html=html,
                    session=session,
                    tournament=tournament,
                    players=players,
                )

                if not match_and_players:
                    logger.info(f"No matches found for {url}")
                    continue
                all_match_and_players.extend(match_and_players)

            all_matches = set()
            all_players = set()
            for match, player1, player2 in all_match_and_players:
                all_matches.add(match)
                all_players.add(player1)
                all_players.add(player2)

            insert_if_not_exists(table=Match, instances=list(all_matches))
            insert_if_not_exists(table=Player, instances=list(all_players))
            logger.success(f"Tournament {tournament.name} matches scraped and stored")


@app.command()
def scrap_matches(
    from_year: int = typer.Option(1990, "--from", help="Start year (default: 1990)"),
    to_year: int = typer.Option(
        datetime.datetime.now().year,
        "--to",
        help="End year (default: current year)",
    ),
) -> None:
    tournaments = get_table(Tournament)
    tournaments = list(
        filter(
            lambda t: from_year <= t.date.year <= to_year,
            tournaments,
        )
    )
    asyncio.run(run_scraping(tournaments))


if __name__ == "__main__":
    app()

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

url_to_players: dict[str, Optional[Player]] = {}


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


def parse_odds(tr1) -> tuple[float, float]:
    tds = tr1.find_all("td", class_="course")
    player_1_odd = tds[0].get_text(strip=True).replace(",", ".")
    player_2_odd = tds[1].get_text(strip=True).replace(",", ".")
    player_1_odd = float(player_1_odd) if player_1_odd.isdigit() else 0.0
    player_2_odd = float(player_2_odd) if player_2_odd.isdigit() else 0.0
    return player_1_odd, player_2_odd


async def player_from_url(
    player_name: str,
    player_url: str,
    session: aiohttp.ClientSession,
) -> Optional[Player]:
    if player_url in url_to_players:
        logger.info(f"Using cached player data for {player_name} from {player_url}")
        return url_to_players[player_url]

    player = await fetch_player(
        session=session,
        player_name=player_name,
        player_detail_url_extension=player_url,
    )

    url_to_players[player_url] = player
    return player


def extract_one_match_data(row1, row2, tournament: Tournament) -> dict[str, Any]:
    (player1_name, player1_url), (player2_name, player2_url) = parse_players_names(
        row1, row2
    )

    date = parse_date(row1, tournament)
    round = parse_round(row1)
    score = parse_score(row1, row2)
    player_1_odds, player_2_odds = parse_odds(row1)

    return {
        "player1_name": player1_name,
        "player1_url": player1_url,
        "player2_name": player2_name,
        "player2_url": player2_url,
        "date": date,
        "round": round,
        "score": score,
        "player_1_odds": player_1_odds,
        "player_2_odds": player_2_odds,
    }


async def extract_matches_and_players(
    html: str,
    session: aiohttp.ClientSession,
    tournament: Tournament,
) -> tuple[list[Match], list[Player]]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", class_="result")
    rows = table.find_all("tr")

    row_one_list = [r for r in rows if r.has_attr("id") and not r["id"].endswith("b")]
    row_two_list = [r for r in rows if r.has_attr("id") and r["id"].endswith("b")]

    assert len(row_one_list) == len(row_two_list), "Mismatch in row counts"
    row_pairs = list(zip(row_one_list, row_two_list))

    match_data_list = []
    players_to_fetch = set()

    for row1, row2 in row_pairs:
        match_data = extract_one_match_data(row1, row2, tournament)
        match_data_list.append(match_data)
        players_to_fetch.add(
            (
                match_data.pop("player1_name"),
                match_data["player1_url"],
            )
        )
        players_to_fetch.add(
            (
                match_data.pop("player2_name"),
                match_data["player2_url"],
            )
        )

    players_in_tournament = await asyncio.gather(
        *[
            player_from_url(
                player_name=name,
                player_url=url,
                session=session,
            )
            for name, url in players_to_fetch
        ]
    )

    tournament_matches = []
    for match_data in match_data_list:
        player_1_url = match_data.pop("player1_url")
        player_2_url = match_data.pop("player2_url")
        player1 = url_to_players.get(player_1_url, None)
        player2 = url_to_players.get(player_2_url, None)

        if player1 is None or player2 is None:
            logger.warning(
                f"At least one player not found: {player_1_url} or {player_2_url}"
            )
            continue

        match = Match(
            tournament_id=tournament.tournament_id,
            player1_id=player1.player_id,
            player2_id=player2.player_id,
            **match_data,
        )
        tournament_matches.append(match)

    return tournament_matches, players_in_tournament


app = typer.Typer()


async def scrap_on_tournament(
    tournament: Tournament, session: aiohttp.ClientSession
) -> None:
    base_url = f"{settings.base_url}/{tournament.match_list_url_extension}"
    logger.info(f"Scraping matches from {tournament.name} : {base_url}")
    extensions = ["/?phase=main", "/?phase=qualification"]

    all_tournament_matches = set()
    all_tournament_players = set()
    for extension in extensions:
        url = f"{base_url}{extension}"  # <-- corrige la concaténation ici

        html: Optional[str] = await get_with_retry(
            session, url, headers={"Accept": "text/html"}
        )

        if html is None:
            logger.error(f"Failed to download {url} after retries")
            continue

        tournament_matches, tournament_players = await extract_matches_and_players(
            html=html,
            session=session,
            tournament=tournament,
        )

        if not tournament_matches:
            logger.info(f"No matches found for {url}")
            continue
        all_tournament_matches.update(tournament_matches)
        all_tournament_players.update(tournament_players)

    insert_if_not_exists(table=Match, instances=list(all_tournament_matches))
    insert_if_not_exists(table=Player, instances=list(all_tournament_players))
    logger.success(f"Tournament {tournament.name} matches scraped and stored")


async def run_scraping(tournaments: list[Tournament]) -> None:
    async with aiohttp.ClientSession() as session:
        players: dict[Any, Any] = {}
        for tournament in tqdm(
            tournaments,
            desc="Scraping tournament matches",
            unit="tournament",
        ):
            await scrap_on_tournament(
                tournament=tournament,
                session=session,
            )


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
            lambda t: t.date is not None and from_year <= t.date.year <= to_year,
            tournaments,
        )
    )
    asyncio.run(run_scraping(tournaments))


if __name__ == "__main__":
    app()

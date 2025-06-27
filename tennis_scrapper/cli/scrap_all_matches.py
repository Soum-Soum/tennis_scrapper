import asyncio
import datetime
from typing import Optional, List

import aiohttp
import typer
from bs4 import BeautifulSoup
from loguru import logger
from sqlmodel import Session, select, col
from tqdm.asyncio import tqdm

from cli.scrap_players import extract_player
from db.db_utils import engine, clear_table
from db.models import Tournament, Gender, Match, Player
from utils.http_utils import async_get_with_retry

app = typer.Typer()


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


def get_default_tournament(
    year: int,
    players_gender: Gender,
    db_session: Session,
) -> Tournament:
    return db_session.exec(
        select(Tournament).where(
            col(Tournament.name).contains("Default"),
            Tournament.year == year,
            Tournament.players_gender == players_gender,
        )
    ).first()


def get_tournament(
    url_extension: Optional[str],
    tournament_name: str,
    year: int,
    players_gender: Gender,
    db_session: Session,
    client_session: aiohttp.ClientSession,
) -> Tournament:
    tournament = db_session.exec(
        select(Tournament).where(
            Tournament.url_extension == url_extension,
            Tournament.year == year,
            Tournament.players_gender == players_gender,
        )
    ).first()
    if tournament is not None:
        return tournament
    else:
        return get_default_tournament(year, players_gender, db_session)


async def get_player_from_url_extension(
    player_url_extension: str, client_session: aiohttp.ClientSession
) -> Player:
    logger.info(f"Scraping player data from {player_url_extension}")
    url = f"https://www.tennisexplorer.com/{player_url_extension}/"
    html = await async_get_with_retry(client_session, url)
    player = extract_player(html=html, player_detail_url_extension=player_url_extension)
    return player


async def scrape_matches_from_url(
    date: datetime.date,
    url: str,
    session: aiohttp.ClientSession,
):
    with Session(engine) as db_session:
        html: Optional[str] = await async_get_with_retry(
            session, url, headers={"Accept": "text/html"}
        )
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", class_="result")

        current_tournament = None

        trs = table.find_all("tr")
        heads = list(
            filter(
                lambda x: x[1].has_attr("class") and "head" in x[1]["class"],
                enumerate(trs),
            )
        )

        players_gender = Gender.MEN if "atp-single" in url else Gender.WOMAN
        row_id_to_tournament = {}
        for row_id, head_row in heads:
            tournament_name = head_row.find("td", class_="t-name").get_text(strip=True)
            tournament_url_extension_a = head_row.find("td", class_="t-name").find("a")
            if not tournament_url_extension_a:
                current_tournament = get_default_tournament(
                    year=date.year, players_gender=players_gender, db_session=db_session
                )
            else:
                tournament_url_extension = tournament_url_extension_a["href"]
                current_tournament = get_tournament(
                    url_extension=tournament_url_extension,
                    tournament_name=tournament_name,
                    year=date.year,
                    players_gender=players_gender,
                    db_session=db_session,
                    client_session=session,
                )

            row_id_to_tournament[row_id] = current_tournament

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

        matches = set()

        for row1, (row_two_id, row2) in row_pairs:
            tournament = row_id_to_tournament[
                max(filter(lambda x: x <= row_two_id, row_id_to_tournament.keys()))
            ]

            try:
                player_1_url_extension = row1.find("td", class_="t-name").find("a")[
                    "href"
                ]
                player_2_url_extension = row2.find("td", class_="t-name").find("a")[
                    "href"
                ]
            except TypeError:
                logger.warning("Unable to find player url -> skip match")
                continue

            player_1_odds_str = (
                row1.find("td", class_="coursew").get_text(strip=True).strip()
            )
            player_2_odds_str = (
                row1.find("td", class_="course").get_text(strip=True).strip()
            )
            player_1_odds = float(player_1_odds_str) if player_1_odds_str else None
            player_2_odds = float(player_2_odds_str) if player_2_odds_str else None

            score = parse_score(row1, row2)

            matches.add(
                Match(
                    tournament_id=tournament.tournament_id,
                    date=date,
                    players_gender=tournament.players_gender,
                    surface=tournament.surface,
                    player_1_url_extension=player_1_url_extension,
                    player_2_url_extension=player_2_url_extension,
                    score=score,
                    player_1_odds=player_1_odds,
                    player_2_odds=player_2_odds,
                )
            )

        db_session.add_all(list(matches))
        db_session.commit()


@app.command()
def scrap_all_matches(
    from_year: int = typer.Option(1990, "--from", help="Start year (default: 1990)"),
    to_year: int = typer.Option(datetime.datetime.now().year, "--to", help="End year"),
    clear_db: bool = typer.Option(
        False, "--clear-db", help="Clear the database before scraping"
    ),
):
    if clear_db:
        logger.info("Clearing the database...")
        clear_table(Match)

    with Session(engine) as db_session:
        max_date_in_db = db_session.exec(
            select(Match.date).order_by(Match.date.desc())
        ).first()

    extensions = ["atp-single", "wta-single"]
    url_template = "https://www.tennisexplorer.com/results/?type={extension}&year={year}&month={month}&day={day}"
    start_date = datetime.date(from_year, 1, 1)
    if max_date_in_db:
        start_date = max(start_date, max_date_in_db + datetime.timedelta(days=1))
    end_date = min(datetime.date(to_year, 12, 31), datetime.date.today())
    logger.info(
        f"Scraping matches from {start_date} to {end_date} for extensions: {extensions}"
    )
    day_diff = (end_date - start_date).days

    async def launch_scraping() -> None:
        semaphore = asyncio.Semaphore(10)

        async def wrapper_scrape(date: datetime, extension: str):
            async with semaphore:
                url = url_template.format(
                    extension=extension,
                    year=date.year,
                    month=date.month,
                    day=date.day,
                )
                async with aiohttp.ClientSession() as session:
                    try:
                        await scrape_matches_from_url(
                            date=date, url=url, session=session
                        )
                    except Exception as e:
                        logger.error(
                            f"Error on scraping for date : {date}. Url : {url}: {e}"
                        )
                        raise e

        tasks = []
        for extension in extensions:
            for day_offset in range(day_diff):
                current_date = start_date + datetime.timedelta(days=day_offset)

                tasks.append(wrapper_scrape(current_date, extension))

        await tqdm.gather(
            *tasks, desc="Scraping matches for every days of the interval", unit="day"
        )

    asyncio.run(launch_scraping())

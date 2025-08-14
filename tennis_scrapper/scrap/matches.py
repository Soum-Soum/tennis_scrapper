import datetime
from loguru import logger


from typing import List, Optional

from tennis_scrapper.db.models import Gender, Match, Tournament

from sqlmodel import Session, col, select


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
    year: int,
    players_gender: Gender,
    db_session: Session,
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


def get_row_id_to_tournament(
    trs: list, players_gender: Gender, date: datetime.date, db_session: Session
) -> dict[int, Tournament]:
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
            current_tournament = get_default_tournament(
                year=date.year, players_gender=players_gender, db_session=db_session
            )
        else:
            tournament_url_extension = tournament_url_extension_a["href"]
            current_tournament = get_tournament(
                url_extension=tournament_url_extension,
                year=date.year,
                players_gender=players_gender,
                db_session=db_session,
            )

        row_id_to_tournament[row_id] = current_tournament

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


def pair_to_match(
    tournament: Tournament,
    row1,
    row2,
    date: datetime.date,
) -> Optional[Match]:

    try:
        player_1_url_extension = row1.find("td", class_="t-name").find("a")["href"]
        player_2_url_extension = row2.find("td", class_="t-name").find("a")["href"]
    except TypeError:
        logger.warning(f"Unable to find player URLs -> skipping row")
        return None

    player_1_odds_str = row1.find("td", class_="coursew").get_text(strip=True).strip()
    player_2_odds_str = row1.find("td", class_="course").get_text(strip=True).strip()
    player_1_odds = float(player_1_odds_str) if player_1_odds_str else None
    player_2_odds = float(player_2_odds_str) if player_2_odds_str else None

    score = parse_score(row1, row2)

    return Match(
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

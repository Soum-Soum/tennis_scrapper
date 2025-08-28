import datetime

from loguru import logger
from sqlmodel import Session, select, text
from tqdm import tqdm

from tennis_scrapper.db.models import Ranking, Gender

ADD_RANKINGS = """
    UPDATE match AS m
    SET
        player_{k}_ranking = r.rank,
        player_{k}_ranking_points  = r.points
    FROM ranking AS r
    WHERE m.players_gender = :gender
        AND m.date >= :start_date AND m.date < :end_date
        AND m.player_{k}_ranking IS NULL
        AND r.circuit = :circuit
        AND r.date    = :start_date
        AND r.player_id = m.player_{k}_id
"""


def add_rankings(db_session: Session) -> None:
    logger.info("Adding ATP points to matches...")

    for gender in Gender:
        # Get all ranking dates for this circuit, sorted
        ranking_dates = db_session.exec(
            select(Ranking.date)
            .where(Ranking.circuit == gender.circuit)
            .group_by(Ranking.date)
            .order_by(Ranking.date)
        ).all()

        logger.info(f"Found {len(ranking_dates)} ranking dates in {gender.circuit}")

        # Build [start, end) windows (prepend epoch)
        distinct_dates = [datetime.date(1970, 1, 1)] + ranking_dates
        date_pairs = list(zip(distinct_dates[:-1], distinct_dates[1:]))

        # Single transaction for this gender
        for start_date, end_date in tqdm(
            date_pairs, desc=f"Adding rankings to matches for {gender.circuit}"
        ):
            params = {
                "gender": gender.value if hasattr(gender, "value") else gender,
                "circuit": gender.circuit,
                "start_date": start_date,
                "end_date": end_date,
            }

            for k in [1, 2]:
                query = text(ADD_RANKINGS.format(k=k))
                query = query.bindparams(**params)
                db_session.exec(query)

        db_session.commit()

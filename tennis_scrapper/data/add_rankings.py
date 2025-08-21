import datetime

from loguru import logger
from sqlmodel import Session, select, text
from tqdm import tqdm

from db.db_utils import engine
from db.models import Match, Ranking, Gender


# def add_atp_points_to_matches(
#     matches: list[Match], rankings: list[Ranking]
# ) -> list[Match]:

#     player_id_to_ranking = {r.player_id: r for r in rankings}

#     matches_with_atp = []
#     for match in matches:
#         player_1_ranking = player_id_to_ranking.get(match.player_1_id, None)
#         player_2_ranking = player_id_to_ranking.get(match.player_2_id, None)

#         updates = {}
#         if player_1_ranking is not None:
#             updates["atp_ranking_player_1"] = int(player_1_ranking.rank)
#             updates["atp_points_player_1"] = int(player_1_ranking.points)

#         if player_2_ranking is not None:
#             updates["atp_ranking_player_2"] = int(player_2_ranking.rank)
#             updates["atp_points_player_2"] = int(player_2_ranking.points)

#         if updates:
#             match = match.model_copy(update=updates)
#         matches_with_atp.append(match)

#     return matches_with_atp


# def add_atp():
#     with Session(engine) as session:

#         for gender in Gender:

#             distinct_dates = list(
#                 session.exec(
#                     select(Ranking.date)
#                     .where(Ranking.circuit == gender.circuit)
#                     .order_by(Ranking.date)
#                     .distinct()
#                 ).all()
#             )

#             distinct_dates = [datetime.date(1970, 1, 1)] + distinct_dates
#             date_pairs = list(zip(distinct_dates[:-1], distinct_dates[1:]))

#             for start_date, end_date in tqdm(
#                 date_pairs,
#                 desc=f"Adding ATP to matches for {gender}",
#                 unit="dates",
#             ):
#                 matches = session.exec(
#                     select(Match).where(
#                         Match.players_gender == gender,
#                         Match.atp_ranking_player_1.is_(None),
#                         Match.atp_ranking_player_2.is_(None),
#                         Match.date >= start_date,
#                         Match.date < end_date,
#                     )
#                 ).all()

#                 rankings = session.exec(
#                     select(Ranking)
#                     .where(Ranking.date == end_date)
#                     .where(Ranking.circuit == gender.circuit)
#                 ).all()

#                 matches = add_atp_points_to_matches(matches, rankings)

#                 session.add_all(matches)
#                 session.commit()

ADD_ATP = """
    UPDATE match AS m
    SET
        atp_ranking_player_{k} = r.rank,
        atp_points_player_{k}  = r.points
    FROM ranking AS r
    WHERE m.players_gender = :gender
        AND m.date >= :start_date AND m.date < :end_date
        AND m.atp_ranking_player_{k} IS NULL
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
        for start_date, end_date in tqdm(date_pairs, desc=f"Adding rankings to matches for {gender.circuit}"):
            params = {
                "gender": gender.value if hasattr(gender, "value") else gender,
                "circuit": gender.circuit,
                "start_date": start_date,
                "end_date": end_date,
            }

            for k in [1, 2]:
                query = text(ADD_ATP.format(k=k))
                query = query.bindparams(**params)
                db_session.exec(query)

        db_session.commit()
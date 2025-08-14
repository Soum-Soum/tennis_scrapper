import datetime

from sqlmodel import Session, select
from tqdm import tqdm

from db.db_utils import engine
from db.models import Match, Ranking, Gender


def add_atp_points_to_matches(
    matches: list[Match], rankings: list[Ranking]
) -> list[Match]:

    player_id_to_ranking = {r.player_id: r for r in rankings}

    matches_with_atp = []
    for match in matches:
        player_1_ranking = player_id_to_ranking.get(match.player_1_id, None)
        player_2_ranking = player_id_to_ranking.get(match.player_2_id, None)

        updates = {}
        if player_1_ranking is not None:
            updates["atp_ranking_player_1"] = int(player_1_ranking.rank)
            updates["atp_points_player_1"] = int(player_1_ranking.points)

        if player_2_ranking is not None:
            updates["atp_ranking_player_2"] = int(player_2_ranking.rank)
            updates["atp_points_player_2"] = int(player_2_ranking.points)

        if updates:
            match = match.model_copy(update=updates)
        matches_with_atp.append(match)

    return matches_with_atp


def add_atp():
    with Session(engine) as session:

        for gender, circuit in [(Gender.MEN, "ATP"), (Gender.WOMAN, "WTA")]:

            distinct_dates = list(
                session.exec(
                    select(Ranking.date)
                    .where(Ranking.circuit == circuit)
                    .order_by(Ranking.date)
                    .distinct()
                ).all()
            )

            distinct_dates = [datetime.date(1970, 1, 1)] + distinct_dates
            date_pairs = list(zip(distinct_dates[:-1], distinct_dates[1:]))

            for start_date, end_date in tqdm(
                date_pairs,
                desc=f"Adding ATP to matches for {gender}",
                unit="dates",
            ):
                matches = session.exec(
                    select(Match).where(
                        Match.players_gender == gender,
                        Match.date >= start_date,
                        Match.date < end_date,
                    )
                ).all()

                rankings = session.exec(
                    select(Ranking)
                    .where(Ranking.date == end_date)
                    .where(Ranking.circuit == circuit)
                ).all()

                matches = add_atp_points_to_matches(matches, rankings)

                session.add_all(matches)
                session.commit()

import datetime

from sqlmodel import Session, select
from tqdm import tqdm

from db.db_utils import engine
from db.models import Match, Ranking, Gender


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

                rankings = session.exec(select(Ranking).where(Ranking.date == end_date))
                player_id_to_ranking = {r.player_id: r for r in rankings}

                for match in matches:
                    player_1_ranking = player_id_to_ranking.get(match.player_1_id, None)
                    player_2_ranking = player_id_to_ranking.get(match.player_2_id, None)

                    if player_1_ranking is not None:
                        match.atp_ranking_player_1 = player_1_ranking.rank
                        match.atp_points_player_1 = player_1_ranking.points

                    if player_2_ranking is not None:
                        match.atp_ranking_player_2 = player_2_ranking.rank
                        match.atp_points_player_2 = player_2_ranking.points

                session.add_all(matches)
                session.commit()

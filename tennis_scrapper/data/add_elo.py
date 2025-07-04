import datetime
from copy import deepcopy

from loguru import logger
from sqlmodel import Session, select, func
from tqdm import tqdm

from db.db_utils import get_table, insert_if_not_exists, engine
from db.models import EloRanking, EloSurface, Player, Surface, Match, Gender

INITIAL_ELO = 1500.0
K = 32


elo_surface_to_surface = {
    EloSurface.CLAY: [Surface.CLAY],
    EloSurface.GRASS: [Surface.GRASS],
    EloSurface.HARD_OR_INDOOR: [Surface.HARD, Surface.INDOOR],
    EloSurface.ALL: [Surface.CLAY, Surface.GRASS, Surface.HARD, Surface.INDOOR],
}

surface_to_elo_surface = {
    Surface.CLAY: EloSurface.CLAY,
    Surface.GRASS: EloSurface.GRASS,
    Surface.HARD: EloSurface.HARD_OR_INDOOR,
    Surface.INDOOR: EloSurface.HARD_OR_INDOOR,
}


def get_updated_elo(match: Match, player_id_to_elo_ranking: dict[str, float], k: float):
    winner_id = match.player_1_id
    loser_id = match.player_2_id

    winner_elo = player_id_to_elo_ranking.get(winner_id, INITIAL_ELO)
    loser_elo = player_id_to_elo_ranking.get(loser_id, INITIAL_ELO)

    # Calculate expected scores using Elo formula
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 - expected_winner

    # Actual results: 1 for winner, 0 for loser
    S_winner, S_loser = 1, 0

    # Update Elo ratings
    winner_elo_new = winner_elo + k * (S_winner - expected_winner)
    loser_elo_new = loser_elo + k * (S_loser - expected_loser)

    # Optionally, return the new ratings
    return winner_elo_new, loser_elo_new


class EloAggregator:

    def __init__(self, players: list[Player]):
        player_id_to_elo_ranking = {player.player_id: INITIAL_ELO for player in players}
        self.surface_to_elo = {
            EloSurface.CLAY: deepcopy(player_id_to_elo_ranking),
            EloSurface.GRASS: deepcopy(player_id_to_elo_ranking),
            EloSurface.HARD_OR_INDOOR: deepcopy(player_id_to_elo_ranking),
            EloSurface.ALL: deepcopy(player_id_to_elo_ranking),
        }

    def update_elo(self, match: Match) -> Match:
        match.player_1_elo = self.surface_to_elo[EloSurface.ALL].get(match.player_1_id)
        match.player_2_elo = self.surface_to_elo[EloSurface.ALL].get(match.player_2_id)

        if match.player_1_elo is None or match.player_2_elo is None:
            logger.warning(
                f"Missing ELO for players {match.player_1_id} or {match.player_2_id} on match {match.match_id}"
            )
            return match

        winner_new_elo, loser_new_elo = get_updated_elo(
            match, self.surface_to_elo[EloSurface.ALL], K
        )
        self.surface_to_elo[EloSurface.ALL][match.player_1_id] = winner_new_elo
        self.surface_to_elo[EloSurface.ALL][match.player_2_id] = loser_new_elo

        if match.surface is Surface.UNKNOWN:
            return match

        elo_surface = surface_to_elo_surface[match.surface]
        match.player_1_elo_on_surface = self.surface_to_elo[elo_surface][
            match.player_1_id
        ]
        match.player_2_elo_on_surface = self.surface_to_elo[elo_surface][
            match.player_2_id
        ]
        winner_new_elo, loser_new_elo = get_updated_elo(
            match, self.surface_to_elo[elo_surface], K
        )
        self.surface_to_elo[elo_surface][match.player_1_id] = winner_new_elo
        self.surface_to_elo[elo_surface][match.player_2_id] = loser_new_elo

        return match


def add_elo():

    with Session(engine) as session:
        min_date, max_date = session.exec(
            select(
                func.min(Match.date).label("date_min"),
                func.max(Match.date).label("date_max"),
            )
        ).first()
        days_diff = (max_date - min_date).days

        players = get_table(Player)

        elo_aggregator = EloAggregator(players=players)

        player_id_to_elo_ranking = {player.player_id: INITIAL_ELO for player in players}

        for days_diff in tqdm(
            range(0, days_diff, 7),
            desc=f"Computing ELOs between {min_date} and {max_date}",
            unit="weeks",
        ):
            current_date = min_date + datetime.timedelta(days=days_diff)
            next_date = current_date + datetime.timedelta(days=7)

            logger.info(f"Processing matches from {current_date} to {next_date}")

            matches = session.exec(
                select(Match)
                .where(
                    Match.date >= current_date,
                    Match.date < next_date,
                )
                .order_by(Match.date),
            ).all()

            matches = [elo_aggregator.update_elo(match) for match in matches]
            session.add_all(matches)
            session.commit()

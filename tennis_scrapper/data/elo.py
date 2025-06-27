import datetime

from loguru import logger
from tqdm import tqdm

from db.db_utils import get_table, insert_if_not_exists
from db.models import EloRanking, EloSurface, Player, Surface, Match

INITIAL_ELO = 1500
K = 32


elo_surface_to_surface = {
    EloSurface.CLAY: [Surface.CLAY],
    EloSurface.GRASS: [Surface.GRASS],
    EloSurface.HARD_OR_INDOOR: [Surface.HARD, Surface.INDOOR],
    EloSurface.ALL: [Surface.CLAY, Surface.GRASS, Surface.HARD, Surface.INDOOR],
}


def get_updated_elo(
    match: Match, player_id_to_elo_ranking: dict[str, float], K: float = 32
):
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
    winner_elo_new = winner_elo + K * (S_winner - expected_winner)
    loser_elo_new = loser_elo + K * (S_loser - expected_loser)

    # Save updated ratings back to the dictionary
    player_id_to_elo_ranking[winner_id] = winner_elo_new
    player_id_to_elo_ranking[loser_id] = loser_elo_new

    # Optionally, return the new ratings
    return winner_elo_new, loser_elo_new


def compute_elos(
    matches: list[Match],
    players: list[Player],
    K: float,
    initial_elo: float,
    elo_surface: EloSurface,
) -> list[float]:

    filtered_matches = list(
        filter(lambda m: m.surface in elo_surface_to_surface[elo_surface], matches)
    )

    player_id_to_elo_ranking = {player.player_id: initial_elo for player in players}

    all_elos = []
    for match in tqdm(
        filtered_matches,
        desc=f"Computing ELOs for surface {elo_surface} on {len(filtered_matches)} matches",
        unit="matches",
    ):
        winner_new_elo, loser_new_elo = get_updated_elo(
            match, player_id_to_elo_ranking, K
        )
        update_date = (match.date + datetime.timedelta(days=1)).date()
        all_elos.extend(
            [
                EloRanking(
                    player_id=match.player_1_id,
                    elo_point=winner_new_elo,
                    date=update_date,
                    surface=elo_surface,
                ),
                EloRanking(
                    player_id=match.player_2_id,
                    elo_point=loser_new_elo,
                    date=update_date,
                    surface=elo_surface,
                ),
            ]
        )

    return all_elos


def add_elo_rating():
    matches = get_table(Match)
    matches = sorted(matches, key=lambda m: m.date)
    players = get_table(Player)

    for elo_surface in EloSurface:
        elos_for_surface = compute_elos(
            matches=matches,
            players=players,
            K=K,
            initial_elo=INITIAL_ELO,
            elo_surface=elo_surface,
        )
        logger.info(f"Adding {len(elos_for_surface)} ELOs for surface {elo_surface}.")
        insert_if_not_exists(EloRanking, instances=elos_for_surface)

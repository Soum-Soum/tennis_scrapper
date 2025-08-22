from datetime import date
import numpy as np
from tennis_scrapper.db.models import Match, Player


def is_winner(match: Match, player_id: str) -> bool:
    return match.player_1_id == player_id


def parse_score(score: str) -> list[tuple[int, int]]:
    return [tuple(map(int, s.split("-"))) for s in score.split()]


def _self_opp(match: Match, player_id: str, attr_p1: str, attr_p2: str):
    """Return (self_value, opp_value) for a pair of attributes on `match`."""
    a1, a2 = getattr(match, attr_p1), getattr(match, attr_p2)
    is_p1 = match.player_1_id == player_id
    return (a1, a2) if is_p1 else (a2, a1)


def _games_lists(match: Match, player_id: str) -> tuple[list[int], list[int]]:
    """Return (games_won_per_set, games_conceded_per_set) for player_id."""
    sets = parse_score(match.score)
    won_idx, conc_idx = (0, 1) if is_winner(match, player_id) else (1, 0)
    return [s[won_idx] for s in sets], [s[conc_idx] for s in sets]


def get_games_won(match: Match, player_id: str) -> list[int]:
    """Get games won by player in each set."""
    won, _ = _games_lists(match, player_id)
    return won

def get_games_conceded(match: Match, player_id: str) -> list[int]:
    """Get games conceded by player in each set."""
    _, conc = _games_lists(match, player_id)
    return conc

def get_elo(match: Match, player_id: str) -> float:
    """Get player's ELO rating for a match."""
    self_elo, _ = _self_opp(match, player_id, "player_1_elo", "player_2_elo")
    return self_elo

def get_opponent_elo(match: Match, player_id: str) -> float:
    """Get opponent's ELO rating for a match."""
    _, opp_elo = _self_opp(match, player_id, "player_1_elo", "player_2_elo")
    return opp_elo

def get_ranking(match: Match, player_id: str) -> float:
    """Get player's ranking for a match."""
    self_rank, _ = _self_opp(match, player_id, "atp_ranking_player_1", "atp_ranking_player_2")
    return self_rank


def get_opponent_ranking(match: Match, player_id: str) -> float:
    """Get opponent's ranking for a match."""
    _, opp_rank = _self_opp(match, player_id, "atp_ranking_player_1", "atp_ranking_player_2")
    return opp_rank


def safe_mean(arr: np.ndarray | list, default=0.0):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return np.mean(arr).item() if arr.size > 0 else default


def compute_player_age(player: Player, date: date) -> float:
    """Calculate player's age at a specific date."""
    return (date - player.birth_date).days / 365.25
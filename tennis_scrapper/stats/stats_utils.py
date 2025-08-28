from datetime import date
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
    self_rank, _ = _self_opp(match, player_id, "player_1_ranking", "player_2_ranking")
    return self_rank


def get_opponent_ranking(match: Match, player_id: str) -> float:
    """Get opponent's ranking for a match."""
    _, opp_rank = _self_opp(match, player_id, "player_1_ranking", "player_2_ranking")
    return opp_rank


def safe_mean(arr: list, default=0.0):
    return sum(arr) / len(arr) if len(arr) > 0 else default


def compute_player_age(player: Player, date: date) -> float:
    """Calculate player's age at a specific date."""
    return (date - player.birth_date).days / 365.25


def is_match_valid(match: Match) -> bool:
    def validate_score_str(score_str: str) -> bool:
        allowed_chars = set("0123456789- ")
        return all(char in allowed_chars for char in score_str)

    if not match.score or not validate_score_str(match.score):
        return False
    return True


def add_key_prefix_suffix(d: dict, prefix: str = "", suffix: str = "") -> dict:
    if prefix != "" and not prefix.endswith("_"):
        prefix += "_"

    if suffix != "" and not suffix.startswith("_"):
        suffix = "_" + suffix

    return {f"{prefix}{key}{suffix}": value for key, value in d.items()}

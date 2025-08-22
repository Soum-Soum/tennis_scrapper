from datetime import date, timedelta
from typing import Dict, Optional
from tennis_scrapper.cli.generate_stats import is_match_sorted
from tennis_scrapper.db.models import Match
from tennis_scrapper.stats.stats_utils import get_elo, get_games_conceded, get_games_won, get_opponent_elo, get_opponent_ranking, is_winner, safe_mean


import numpy as np


def compute_opponent_elo_stat(
    selected_matches: list[Match], player_id: str
) -> dict[str, float]:
    opponent_elo_winnig_match = []
    opponent_elo_losing_match = []
    opponent_ranking_winnig_match = []
    opponent_ranking_losing_match = []

    for match in selected_matches:
        opponent_elo = get_opponent_elo(match, player_id)
        opponent_ranking = get_opponent_ranking(match, player_id)
        if opponent_elo is None:
            continue


        if is_winner(match, player_id):
            opponent_elo_winnig_match.append(opponent_elo)
        else:
            opponent_elo_losing_match.append(opponent_elo)

        if opponent_ranking is None:
            continue

        if is_winner(match, player_id):
            opponent_ranking_winnig_match.append(opponent_ranking)
        else:
            opponent_ranking_losing_match.append(opponent_ranking)

    mean_elo_opponent_winning_match = safe_mean(np.array(opponent_elo_winnig_match))
    mean_elo_opponent_losing_match = safe_mean(np.array(opponent_elo_losing_match))
    mean_ranking_opponent_winning_match = safe_mean(np.array(opponent_ranking_winnig_match))
    mean_ranking_opponent_losing_match = safe_mean(np.array(opponent_ranking_losing_match))

    return {
        "mean_elo_opponent_winning_match": mean_elo_opponent_winning_match,
        "mean_elo_opponent_losing_match": mean_elo_opponent_losing_match,
        "mean_ranking_opponent_winning_match": mean_ranking_opponent_winning_match,
        "mean_ranking_opponent_losing_match": mean_ranking_opponent_losing_match,
    }


def compute_match_played_stats(matches: list[Match], match_date: date):
    deltas = {
        "week": timedelta(weeks=1),
        "month": timedelta(weeks=4),
        "trimester": timedelta(weeks=13),
        "year": timedelta(weeks=52),
    }

    min_date = min((match.date for match in matches))
    days_diff = (match_date - min_date).days

    stats = {}
    for k, v in deltas.items():
        cutoff_date = match_date - v
        matches_in_period = list(filter(lambda x: x.date >= cutoff_date, matches))
        stats[f"match_played_last_{k}"] = len(matches_in_period)
        stats[f"match_per_day_last_{k}"] = stats[f"match_played_last_{k}"] / days_diff

    return stats


def compute_h2h_stats(
    matches: list[Match], player_1_id: str, player_2_id: str
) -> Dict[str, float]:
    """Compute head-to-head statistics between two players."""
    stats = {}
    matches = sorted(matches, key=lambda x: x.date)

    player_1_wins = np.sum([is_winner(match, player_1_id) for match in matches]).item()
    player_2_wins = np.sum([is_winner(match, player_2_id) for match in matches]).item()

    stats["h2h_player_1_wins"] = player_1_wins
    stats["h2h_player_2_wins"] = player_2_wins
    stats["h2h_player_1_win_rate"] = player_1_wins / len(matches) if matches else 0.0
    stats["h2h_player_2_win_rate"] = player_2_wins / len(matches) if matches else 0.0
    stats["h2h_total_matches"] = len(matches)

    return stats


def compute_player_match_based_stats(
    matches: list[Match], player_id: str, k: Optional[list[int]]
) -> Dict[str, float]:
    """Compute comprehensive player statistics for different match windows."""

    stats = {}
    assert is_match_sorted(matches), "Matches must be sorted by date"

    for k_value in k:
        selected_matches = matches[-k_value:]
        if len(selected_matches) == 0:
            continue

        all_games_won = sum(
            [get_games_won(match, player_id) for match in selected_matches], start=[]
        )
        all_games_conceded = sum(
            [get_games_conceded(match, player_id) for match in selected_matches],
            start=[],
        )

        games_won_by_set = safe_mean(np.array(all_games_won))
        games_conceded_by_set = safe_mean(np.array(all_games_conceded))
        winning_rate = safe_mean(
            [is_winner(match, player_id) for match in selected_matches]
        )

        first_elo = get_elo(selected_matches[0], player_id)
        last_elo = get_elo(selected_matches[-1], player_id)

        opponent_stats = compute_opponent_elo_stat(selected_matches, player_id)
        opponent_stats = {f"{key}@k={k_value}": value for key, value in opponent_stats.items()}
        stats.update(opponent_stats)

        stats[f"elo_diff_@k={k_value}"] = last_elo - first_elo
        stats[f"win_rate_@k={k_value}"] = winning_rate
        stats[f"games_won_@k={k_value}"] = games_won_by_set
        stats[f"games_conceded_@k={k_value}"] = games_conceded_by_set

    return stats
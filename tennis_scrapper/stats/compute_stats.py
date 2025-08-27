from datetime import date, timedelta
from typing import Dict, Optional
from db.models import Match, Player
from stats.stats_utils import (
    add_key_prefix_suffix,
    compute_player_age,
    get_elo,
    get_games_conceded,
    get_games_won,
    get_opponent_elo,
    get_opponent_ranking,
    is_winner,
    safe_mean,
)


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

    mean_elo_opponent_winning_match = safe_mean(opponent_elo_winnig_match)
    mean_elo_opponent_losing_match = safe_mean(opponent_elo_losing_match)
    mean_ranking_opponent_winning_match = safe_mean(opponent_ranking_winnig_match)
    mean_ranking_opponent_losing_match = safe_mean(opponent_ranking_losing_match)

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
    len_matches = len(matches)

    player_1_wins = sum([is_winner(match, player_1_id) for match in matches])
    player_2_wins = len_matches - player_1_wins

    stats["h2h_player_1_wins"] = player_1_wins
    stats["h2h_player_2_wins"] = player_2_wins
    stats["h2h_total_matches"] = len_matches
    if len_matches > 0:
        stats["h2h_player_1_win_rate"] = player_1_wins / len_matches
        stats["h2h_player_2_win_rate"] = player_2_wins / len_matches
    else:
        stats["h2h_player_1_win_rate"] = 0
        stats["h2h_player_2_win_rate"] = 0

    return stats


def is_match_sorted(matches: list[Match]) -> bool:
    """Check if matches are sorted by date."""
    return all(matches[i].date <= matches[i + 1].date for i in range(len(matches) - 1))


def compute_player_match_based_stats(
    matches: list[Match], player_id: str, k: Optional[list[int]]
) -> Dict[str, float]:
    """Compute comprehensive player statistics for different match windows."""

    assert is_match_sorted(matches), "Matches must be sorted by date"
    stats = {}

    for k_value in k:
        selected_matches = matches[-k_value:]
        if not selected_matches:
            continue

        # Pré-calculer les infos en un seul passage
        games_won_all, games_conceded_all, winners = [], [], []
        for m in selected_matches:
            games_won_all.extend(get_games_won(m, player_id))
            games_conceded_all.extend(get_games_conceded(m, player_id))
            winners.append(is_winner(m, player_id))

        # Moyennes (safe_mean peut accepter une liste directement)
        games_won_by_set = safe_mean(games_won_all)
        games_conceded_by_set = safe_mean(games_conceded_all)
        winning_rate = safe_mean(winners)

        # ELO
        first_elo = get_elo(selected_matches[0], player_id)
        last_elo = get_elo(selected_matches[-1], player_id)

        # Stats adversaires
        opponent_stats = compute_opponent_elo_stat(selected_matches, player_id)
        opponent_stats = {f"{key}_@k={k_value}": v for key, v in opponent_stats.items()}
        stats.update(opponent_stats)

        # Résultats
        stats[f"elo_diff_@k={k_value}"] = last_elo - first_elo
        stats[f"win_rate_@k={k_value}"] = winning_rate
        stats[f"games_won_@k={k_value}"] = games_won_by_set
        stats[f"games_conceded_@k={k_value}"] = games_conceded_by_set

    return stats


def compute_one_player_stat(
    matches: list[Match],
    matches_on_surface: list[Match],
    match: Match,
    player: Player,
    ks: list[int],
):
    player_stats = compute_player_match_based_stats(matches, player.player_id, k=ks)
    player_stats = {f"{key}": value for key, value in player_stats.items()}

    player_stats_on_surface = compute_player_match_based_stats(
        matches_on_surface, player.player_id, k=ks
    )
    player_stats_on_surface = add_key_prefix_suffix(
        player_stats_on_surface, suffix="_on_surface"
    )

    match_played_stats = compute_match_played_stats(matches, match.date)

    return {
        "age": compute_player_age(player, match.date),
        **player_stats,
        **player_stats_on_surface,
        **match_played_stats,
    }


def compute_one_match_stat(
    match: Match,
    player_1: Player,
    player_2: Player,
    matches_player_1: list[Match],
    matches_player_1_on_surface: list[Match],
    matches_player_2: list[Match],
    matches_player_2_on_surface: list[Match],
    h2h_matches: list[Match],
    ks: list[int],
) -> dict:
    player_1_stats = compute_one_player_stat(
        matches=matches_player_1,
        matches_on_surface=matches_player_1_on_surface,
        player=player_1,
        match=match,
        ks=ks,
    )
    player_1_stats = add_key_prefix_suffix(player_1_stats, prefix="player_1")

    player_2_stats = compute_one_player_stat(
        matches=matches_player_2,
        matches_on_surface=matches_player_2_on_surface,
        player=player_2,
        match=match,
        ks=ks,
    )
    player_2_stats = add_key_prefix_suffix(player_2_stats, prefix="player_2")

    h2h_stats = compute_h2h_stats(
        h2h_matches, player_1_id=match.player_1_id, player_2_id=match.player_2_id
    )

    # Combine all data
    return {
        **match.model_dump(exclude={"date"}),
        "date": str(match.date),
        "player_1_history_size": len(matches_player_1),
        "player_2_history_size": len(matches_player_2),
        "player_1_history_size_on_surface": len(matches_player_1_on_surface),
        "player_2_history_size_on_surface": len(matches_player_2_on_surface),
        **player_1_stats,
        **player_2_stats,
        **h2h_stats,
    }

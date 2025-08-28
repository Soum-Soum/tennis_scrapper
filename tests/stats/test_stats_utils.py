from datetime import date
import numpy as np

import pytest

from tennis_scrapper.stats.stats_utils import (
    parse_score,
    get_games_won,
    get_games_conceded,
    get_elo,
    get_opponent_elo,
    get_ranking,
    get_opponent_ranking,
    safe_mean,
    compute_player_age,
    is_winner,
    is_match_valid,
    add_key_prefix_suffix,
)


class DummyMatch:
    def __init__(
        self,
        player_1_id: str,
        player_2_id: str,
        score: str,
        player_1_elo: float = 1500.0,
        player_2_elo: float = 1400.0,
        player_1_ranking: int = 1,
        player_2_ranking: int = 2,
    ):
        self.player_1_id = player_1_id
        self.player_2_id = player_2_id
        self.score = score
        self.player_1_elo = player_1_elo
        self.player_2_elo = player_2_elo
        self.player_1_ranking = player_1_ranking
        self.player_2_ranking = player_2_ranking
        # Optional attributes to mimic db.models.Match interface when needed by validators
        self.player_1_url_extension = "p1"
        self.player_2_url_extension = "p2"


class DummyPlayer:
    def __init__(self, birth_date: date):
        self.birth_date = birth_date


def test_parse_score_simple():
    s = "6-3 7-5"
    parsed = parse_score(s)
    assert parsed == [(6, 3), (7, 5)]


def test_games_won_and_conceded_for_winner():
    m = DummyMatch("p1", "p2", "6-3 7-5")
    won = get_games_won(m, "p1")
    conc = get_games_conceded(m, "p1")
    assert won == [6, 7]
    assert conc == [3, 5]


def test_games_won_and_conceded_for_loser():
    m = DummyMatch("p1", "p2", "6-3 7-5")
    won = get_games_won(m, "p2")
    conc = get_games_conceded(m, "p2")
    assert won == [3, 5]
    assert conc == [6, 7]


def test_elo_and_ranking_accessors():
    m = DummyMatch(
        "p1",
        "p2",
        "6-4 4-6 7-6",
        player_1_elo=1600,
        player_2_elo=1550,
        player_1_ranking=10,
        player_2_ranking=20,
    )
    assert get_elo(m, "p1") == 1600
    assert get_opponent_elo(m, "p1") == 1550
    assert get_ranking(m, "p2") == 20
    assert get_opponent_ranking(m, "p2") == 10


def test_safe_mean_and_empty():
    arr = np.array([1.0, 2.0, 3.0])
    assert safe_mean(arr) == pytest.approx(2.0)
    assert safe_mean([]) == 0.0


def test_compute_player_age():
    birth = date(2000, 1, 1)
    player = DummyPlayer(birth)
    on_date = date(2020, 1, 1)
    age = compute_player_age(player, on_date)
    assert pytest.approx(age, rel=1e-3) == (on_date - birth).days / 365.25


def test_is_winner():
    m = DummyMatch("p1", "p2", "6-3 7-5")
    assert is_winner(m, "p1") is True
    assert is_winner(m, "p2") is False


def test_is_match_valid_true_and_false():
    # valid score
    m_valid = DummyMatch("p1", "p2", "6-4 6-4")
    assert is_match_valid(m_valid) is True
    # invalid score (letters)
    m_invalid_chars = DummyMatch("p1", "p2", "6-a 6-4")
    assert is_match_valid(m_invalid_chars) is False
    # invalid score (empty)
    m_empty = DummyMatch("p1", "p2", "")
    assert is_match_valid(m_empty) is False


def test_add_key_prefix_suffix():
    base = {"a": 1, "b": 2}
    assert add_key_prefix_suffix(base, prefix="pre") == {"pre_a": 1, "pre_b": 2}
    assert add_key_prefix_suffix(base, suffix="suf") == {"a_suf": 1, "b_suf": 2}
    assert add_key_prefix_suffix(base, prefix="pre", suffix="suf") == {
        "pre_a_suf": 1,
        "pre_b_suf": 2,
    }

from datetime import date
import numpy as np

import pytest

from stats.stats_utils import parse_score, get_games_won, get_games_conceded, get_elo, get_opponent_elo, get_ranking, get_opponent_ranking, safe_mean, compute_player_age 


class DummyMatch:
	def __init__(
		self,
		player_1_id: str,
		player_2_id: str,
		score: str,
		player_1_elo: float = 1500.0,
		player_2_elo: float = 1400.0,
		atp_ranking_player_1: int = 1,
		atp_ranking_player_2: int = 2,
	):
		self.player_1_id = player_1_id
		self.player_2_id = player_2_id
		self.score = score
		self.player_1_elo = player_1_elo
		self.player_2_elo = player_2_elo
		self.atp_ranking_player_1 = atp_ranking_player_1
		self.atp_ranking_player_2 = atp_ranking_player_2


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
	m = DummyMatch("p1", "p2", "6-4 4-6 7-6", player_1_elo=1600, player_2_elo=1550, atp_ranking_player_1=10, atp_ranking_player_2=20)
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

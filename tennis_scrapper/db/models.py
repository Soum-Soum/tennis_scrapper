import datetime
import hashlib
from enum import StrEnum
from typing import Literal, Self

from loguru import logger
from sqlmodel import Field, SQLModel


class Surface(StrEnum):
    INDOOR = "Indoors"
    HARD = "Hard"
    CLAY = "Clay"
    GRASS = "Grass"
    UNKNOWN = "Unknown"


class Gender(StrEnum):
    MEN = "MEN"
    WOMAN = "WOMAN"

    @classmethod
    def from_string(cls, value: str) -> Self:
        value = value.strip().upper()
        if value in ["MAN", "MEN"]:
            return cls.MEN
        elif value in ["WOMAN", "WOMEN"]:
            return cls.WOMAN
        raise ValueError(f"Invalid gender value: {value}")


class HashedIDModel(SQLModel):
    @staticmethod
    def generate_hashed_id(*args) -> str:
        s = "".join(str(arg) for arg in args)
        return hashlib.md5(s.encode("utf-8")).hexdigest()


class Tournament(HashedIDModel, table=True):
    tournament_id: str = Field(default=None, primary_key=True)
    name: str = Field(nullable=False, description="Tournament name")
    date: datetime.date = Field(description="Tournament date in format", nullable=True)
    surface: Surface = Field(nullable=False, description="Tournament surface type")
    cash_prize: float = Field(nullable=False, description="Cash prize in USD")
    players_gender: Gender = Field(
        nullable=False, description="Is tournament for men or for women"
    )
    match_list_url_extension: str = Field(
        nullable=False, description="URL extension for match list"
    )
    has_been_scraped: bool = Field(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.tournament_id:
            raise ValueError(f"Tournament ID is already set: {self.tournament_id}")
        self.tournament_id = self.generate_hashed_id(
            self.name,
            self.date,
            self.surface,
            self.cash_prize,
            self.players_gender,
            self.match_list_url_extension,
        )

    def __hash__(self):
        return int(self.tournament_id, 16)


class Player(HashedIDModel, table=True):
    player_id: str = Field(default=None, primary_key=True)
    name: str = Field(nullable=False)
    country: str = Field(nullable=False)
    birth_date: datetime.date = Field(nullable=False)
    gender: Gender = Field(nullable=False)
    preferred_hand: str = Field(nullable=False, description="Player's preferred hand")
    player_detail_url_extension: str = Field(
        description="URL extension for player details"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.player_id:
            raise ValueError(f"Player ID is already set: {self.player_id}")
        self.player_id = self.generate_hashed_id(
            self.name,
            self.country,
            self.birth_date,
            self.gender,
            self.preferred_hand,
        )

    def __hash__(self):
        return int(self.player_id, 16)


class Match(HashedIDModel, table=True):
    match_id: str = Field(default=None, primary_key=True)
    tournament_id: str = Field(foreign_key="tournament.tournament_id", nullable=False)
    date: datetime.datetime = Field(nullable=False)
    circuit: str = Field(nullable=False, description="ATP or WTA circuit")
    round: str = Field(
        nullable=True,
        description="Round of the match (e.g. 'Final', 'Semi-Final', 'Quarter-Final')",
    )
    surface: Surface = Field(nullable=False, description="Match surface type")
    player_1_id: str = Field(foreign_key="player.player_id", nullable=False)
    player_2_id: str = Field(foreign_key="player.player_id", nullable=False)
    score: str = Field(
        nullable=False, description="Match score in format '6-3 6-4' or '6-3 3-6 6-4'"
    )
    player_1_odds: float = Field(
        nullable=False, description="Odds for player 1 to win the match"
    )
    player_2_odds: float = Field(
        nullable=False, description="Odds for player 2 to win the match"
    )
    player_1_elo: float = Field(
        default=None,
        nullable=True,
        description="Elo rating of player 1 before the match",
    )
    player_2_elo: float = Field(
        default=None,
        nullable=True,
        description="Elo rating of player 2 before the match",
    )
    player_1_elo_on_surface: float = Field(
        default=None,
        nullable=True,
        description="Elo rating of player 1 on the match surface before the match",
    )
    player_2_elo_on_surface: float = Field(
        default=None,
        nullable=True,
        description="Elo rating of player 2 on the match surface before the match",
    )
    atp_ranking_player_1: int = Field(
        default=None,
        nullable=True,
        description="ATP ranking of player 1 before the match",
    )
    atp_ranking_player_2: int = Field(
        default=None,
        nullable=True,
        description="ATP ranking of player 2 before the match",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.match_id:
            raise ValueError(f"Match ID is already set: {self.match_id}")
        self.match_id = self.generate_hashed_id(
            self.tournament_id,
            self.date,
            self.player_1_id,
            self.player_2_id,
            self.score,
            self.player_1_odds,
            self.player_2_odds,
            self.round,
            self.surface,
        )

    def __hash__(self):
        return int(self.match_id, 16)


class Ranking(HashedIDModel, table=True):
    ranking_id: str = Field(default=None, primary_key=True)
    date: datetime.date
    rank: float
    player_name: str
    player_detail_url_extension: str
    points: float
    circuit: str = Field(
        nullable=False, description="ATP or WTA circuit"
    )
    player_id: str = Field(
        default=None,
        foreign_key="player.player_id",
        nullable=True,
        description="Player ID for the ranking",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.ranking_id:
            raise ValueError(f"Ranking ID is already set: {self.ranking_id}")
        self.ranking_id = self.generate_hashed_id(
            self.date,
            self.rank,
            self.player_name,
            self.player_detail_url_extension,
            self.points,
            self.circuit
        )


class EloSurface(StrEnum):
    CLAY = "CLAY"
    GRASS = "GRASS"
    HARD_OR_INDOOR = "HARD_OR_INDOOR"
    ALL = "ALL"


class EloRanking(HashedIDModel, table=False):
    elo_ranking_id: str = Field(default=None, primary_key=True)
    player_id: str
    elo_point: float
    date: datetime.date
    surface: EloSurface

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.elo_ranking_id:
            raise ValueError(f"Elo ranking ID is already set: {self.elo_ranking_id}")
        self.elo_ranking_id = self.generate_hashed_id(
            self.player_id, self.elo_point, self.date, self.surface
        )

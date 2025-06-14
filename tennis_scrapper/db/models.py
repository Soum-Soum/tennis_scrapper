import datetime
import hashlib
from enum import StrEnum
from typing import Self

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
    player1_id: str = Field(foreign_key="player.player_id", nullable=False)
    player2_id: str = Field(foreign_key="player.player_id", nullable=False)

    score: str = Field(
        nullable=False, description="Match score in format '6-3 6-4' or '6-3 3-6 6-4'"
    )
    player_1_odds: float = Field(
        nullable=False, description="Odds for player 1 to win the match"
    )
    player_2_odds: float = Field(
        nullable=False, description="Odds for player 2 to win the match"
    )
    round: str = Field(
        nullable=True,
        description="Round of the match (e.g. 'Final', 'Semi-Final', 'Quarter-Final')",
    )
    surface: Surface = Field(nullable=False, description="Match surface type")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.match_id:
            raise ValueError(f"Match ID is already set: {self.match_id}")
        self.match_id = self.generate_hashed_id(
            self.tournament_id,
            self.date,
            self.player1_id,
            self.player2_id,
            self.score,
            self.player_1_odds,
            self.player_2_odds,
            self.round,
        )

    def __hash__(self):
        return int(self.match_id, 16)

import datetime
from enum import StrEnum
import uuid
from sqlmodel import Field, SQLModel


class Surface(StrEnum):
    INDOOR = "Indoors"
    HARD = "Hard"
    CLAY = "Clay"
    GRASS = "Grass"


class Gender(StrEnum):
    MEN = "MEN"
    WOMEN = "WOMEN"


class PreferedHand(StrEnum):
    RIGHT = "RIGHT"
    LEFT = "LEFT"


class Tournament(SQLModel, table=True):
    tournament_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(nullable=False)
    date: datetime.date = Field(nullable=False)
    surface: Surface = Field(nullable=False)
    cash_prize: float = Field(nullable=False)
    players_gender: Gender = Field(nullable=False)
    match_list_url_extension: str = Field(
        nullable=False, description="URL extension for match list"
    )


class Player(SQLModel, table=True):
    player_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(nullable=False)
    country: str = Field(nullable=False)
    birth_date: datetime.date = Field(nullable=False)
    gender: Gender = Field(nullable=False)
    prefered_hand: PreferedHand = Field(
        nullable=False, description="Player's prefered hand"
    )


class Match(SQLModel, table=True):
    match_id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    tournament_id: uuid.UUID = Field(
        foreign_key="tournament.tournament_id", nullable=False
    )
    date: datetime.datetime = Field(nullable=False)
    player1_id: uuid.UUID = Field(foreign_key="player.player_id", nullable=False)
    player2_id: uuid.UUID = Field(foreign_key="player.player_id", nullable=False)
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

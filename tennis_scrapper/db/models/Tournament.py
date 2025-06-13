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

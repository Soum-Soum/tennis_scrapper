import datetime
from typing import Iterable

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import SQLModel

from db.models import Player, Match, Surface, Gender


async def create_test_async_engine() -> AsyncEngine:
    """Create an async in-memory SQLite engine suitable for tests.

    Using StaticPool ensures the same in-memory database is reused across sessions.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    return engine


async def dispose_engine(engine: AsyncEngine) -> None:
    await engine.dispose()


def make_player(
    name: str,
    url_extension: str,
    *,
    country: str = "XX",
    birth_date: datetime.date = datetime.date(1990, 1, 1),
    gender: Gender = Gender.MEN,
    preferred_hand: str = "R",
) -> Player:
    return Player(
        name=name,
        country=country,
        birth_date=birth_date,
        gender=gender,
        preferred_hand=preferred_hand,
        url_extension=url_extension,
    )


def make_match(
    *,
    date: datetime.date,
    player_1: Player,
    player_2: Player,
    score: str = "6-4 6-4",
    surface: Surface = Surface.HARD,
    players_gender: Gender = Gender.MEN,
    tournament_url_extension: str = "tournament-x",
) -> Match:
    return Match(
        tournament_url_extension=tournament_url_extension,
        date=date,
        players_gender=players_gender,
        surface=surface,
        player_1_id=player_1.player_id,
        player_2_id=player_2.player_id,
        player_1_url_extension=player_1.url_extension,
        player_2_url_extension=player_2.url_extension,
        score=score,
    )


async def seed_players_and_matches(
    *,
    session: AsyncSession,
    players: Iterable[Player],
    matches: Iterable[Match],
) -> None:
    session.add_all(list(players))
    await session.commit()
    session.add_all(list(matches))
    await session.commit()

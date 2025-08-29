from contextlib import contextmanager
from datetime import date
import sys
from typing import Type, TypeVar, Optional

from loguru import logger
from sqlalchemy import Engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import (
    SQLModel,
    Sequence,
    create_engine,
    Session,
    inspect,
    or_,
    select,
    delete,
    text,
)
from tqdm import tqdm

from tennis_scrapper.conf.config import settings
from tennis_scrapper.db.models import (
    Gender,
    Match,
    Ranking,
    Surface,
    Player,
    Tournament,
)

def get_engine() -> Engine:
    try:
        return create_engine(settings.db_url, echo=False)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("If using PostgreSQL, make sure the database is running and accessible.")
        sys.exit(1)


@contextmanager
def get_session(existing_session: Optional[Session] = None):
    if existing_session is not None:
        yield existing_session
    else:
        with Session(get_engine()) as session:
            yield session


def create_db_and_tables():
    SQLModel.metadata.create_all(get_engine())


T = TypeVar("T", bound=SQLModel)


def get_table(table: Type[T]) -> Sequence[T]:
    with Session(get_engine()) as session:
        statement = select(table)
        result = session.exec(statement)
        return result.all()


def clear_table(table: Type[T]) -> None:
    with Session(get_engine()) as db_session:
        db_session.exec(delete(table))
        db_session.commit()


def get_conflict_columns(engine: Engine, table: Type[T]):
    inspector = inspect(engine)
    table_name = table.__tablename__

    unique_constraints = inspector.get_unique_constraints(table_name)
    if unique_constraints:
        columns = unique_constraints[0]["column_names"]
        if columns:
            return columns

    pk_constraint = inspector.get_pk_constraint(table_name)
    pk_columns = pk_constraint.get("constrained_columns", [])
    if pk_columns:
        return pk_columns

    raise ValueError(
        f"No unique or primary key constraint found for table {table_name}"
    )


def insert_if_not_exists(
    db_session: Session, table: Type[T], instances: list[T]
) -> None:
    if len(instances) == 0:
        return

    conflict_columns = get_conflict_columns(db_session.get_bind(), table)

    def insert_one_batch(batch: list[T]) -> None:
        values = [instance.model_dump() for instance in batch]
        stmt = (
            insert(table)
            .values(values)
            .on_conflict_do_nothing(index_elements=conflict_columns)
        )
        db_session.exec(stmt)
        db_session.commit()

    batch_size = 20_000
    if len(instances) <= batch_size:
        insert_one_batch(instances)
    else:
        for i in tqdm(
            range(0, len(instances), batch_size),
            desc=f"Inserting {table.__name__} instances",
            unit="batch",
        ):
            insert_one_batch(instances[i : i + batch_size])


def add_player_id_to_ranking_table(db_session: Session):
    logger.info("Adding player IDs to ranking table...")

    UPDATE_RANKING_TABLE = """
        UPDATE ranking r
        SET player_id = p.player_id
        FROM player p
        WHERE r.player_id IS NULL
        AND lower(trim(r.player_detail_url_extension)) = lower(trim(p.url_extension));
    """

    db_session.exec(text(UPDATE_RANKING_TABLE))
    db_session.commit()


def add_player_id_to_match_table(db_session: Session):
    logger.info("Adding player IDs to match table...")

    UPDATE_MATCH_TABLE = """
        UPDATE match m
        SET player_{k}_id = p.player_id
        FROM player p
        WHERE m.player_{k}_id IS NULL
        AND lower(trim(m.player_{k}_url_extension)) = lower(trim(p.url_extension));
    """

    for k in [1, 2]:
        logger.info(f"Updating player {k} IDs in match table...")
        db_session.exec(text(UPDATE_MATCH_TABLE.format(k=k)))
        db_session.commit()


def add_tournament_to_match_table(db_session: Session):
    logger.info("Adding tournament data to match table...")

    UPDATE_MATCH_TABLE = """
        UPDATE match m
        SET tournament_id = t.tournament_id,
            surface = t.surface
        FROM tournament t
        WHERE m.tournament_id IS NULL
        AND m.tournament_url_extension = t.url_extension;
    """

    db_session.exec(text(UPDATE_MATCH_TABLE))
    db_session.commit()

    match_without_tournament = db_session.exec(
        select(Match).where(Match.tournament_id.is_(None))
    ).all()

    logger.info(
        f"After adding tournament data, {len(match_without_tournament)} matches still have no tournament data."
    )
    defaults_tournament = db_session.exec(
        text(
            """
        SELECT * FROM tournament WHERE name like '%DEFAULT%' 
    """
        )
    )
    logger.info(f"Found {defaults_tournament.rowcount} default tournaments.")
    gender_year_to_tournament = {}

    for tournament in defaults_tournament:
        key = (tournament.players_gender, tournament.year)
        gender_year_to_tournament[key] = tournament

    for match in match_without_tournament:
        key = (match.players_gender, match.date.year)
        if key in gender_year_to_tournament:
            tournament = gender_year_to_tournament[key]
            match.tournament_id = tournament.tournament_id
            match.surface = tournament.surface
            db_session.add(match)

    db_session.commit()


def get_player_by_id(
    player_id: str, db_session: Optional[Session] = None
) -> Optional[Player]:
    """Get a player by ID."""
    with get_session(db_session) as session:
        player = session.exec(
            select(Player).where(Player.player_id == player_id)
        ).first()
        return player


def get_one_player_matches(
    player_id: str,
    date: Optional[date] = None,
    surface: Optional[Surface] = None,
    limit: Optional[int] = None,
    db_session: Optional[Session] = None,
) -> list[Match]:

    with get_session(db_session) as db_session:
        statement = select(Match).where(
            or_(Match.player_1_id == player_id, Match.player_2_id == player_id),
        )
        if date:
            statement = statement.where(Match.date < date)
        if surface:
            statement = statement.where(Match.surface == surface)

        statement = statement.order_by(Match.date.desc())

        if limit:
            statement = statement.limit(limit)

        return db_session.exec(statement).all()


def get_player_by_url_extension(
    url_extension: str, db_session: Optional[Session] = None
) -> Player:
    with get_session(db_session) as session:
        statement = select(Player).where(Player.url_extension == url_extension)
        player = session.exec(statement).first()
        if player is None:
            raise ValueError(f"Player with url extension {url_extension} not found")
        return player


def get_tournament_by_url(
    url: str, db_session: Optional[Session] = None
) -> Optional[Tournament]:
    with get_session(db_session) as db_session:
        return db_session.exec(
            select(Tournament).where(Tournament.url_extension == url)
        ).first()


def get_last_ranking(db_session: Session, gender: Gender) -> list[Ranking]:
    logger.info(f"Fetching last ranking for circuit: {gender.circuit}")

    last_ranking_date = db_session.exec(
        select(Ranking.date)
        .where(Ranking.circuit == gender.circuit)
        .order_by(Ranking.date.desc())
        .limit(1)
    ).first()

    statement = select(Ranking).where(
        Ranking.circuit == gender.circuit, Ranking.date == last_ranking_date
    )
    return db_session.exec(statement).all()


async def get_player_history(
    db_session: AsyncSession,
    player_id: str,
    cutting_date: Optional[date] = None,
    limit: Optional[int] = None,
) -> list[Match]:
    statement = select(Match).where(
        or_(Match.player_1_id == player_id, Match.player_2_id == player_id),
    )
    if cutting_date:
        statement = statement.where(Match.date < cutting_date)
    if limit:
        statement = statement.limit(limit)

    statement = statement.order_by(Match.date)

    result = await db_session.exec(statement)
    return result.all()


async def get_h2h_matches(
    db_session: AsyncSession,
    player_1_id: str,
    player_2_id: str,
    cutting_date: Optional[date] = None,
) -> list[Match]:
    statement = select(Match).where(
        or_(Match.player_1_id == player_1_id, Match.player_2_id == player_1_id),
        or_(Match.player_1_id == player_2_id, Match.player_2_id == player_2_id),
    )
    if cutting_date:
        statement = statement.where(Match.date < cutting_date)

    statement = statement.order_by(Match.date)

    result = await db_session.exec(statement)
    return result.all()

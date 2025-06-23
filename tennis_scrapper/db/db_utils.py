from typing import Type, TypeVar, Optional
import sqlalchemy
from sqlmodel import (
    SQLModel,
    Sequence,
    create_engine,
    Session,
    inspect,
    select,
    delete,
    func,
)
from sqlalchemy.dialects.postgresql import insert
from tqdm import tqdm
import sys

from conf.config import settings
from db.models import Tournament, Player

DB_PATH = settings.db_url


def get_engine():
    try:
        return create_engine(DB_PATH, echo=False)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("If using PostgreSQL, make sure the database is running and accessible.")
        sys.exit(1)


def create_db_and_tables():
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    return engine


def get_session(engine=None):
    if engine is None:
        engine = get_engine()
    return Session(engine)


engine = get_engine()
SQLModel.metadata.create_all(engine)


T = TypeVar("T", bound=SQLModel)


def get_table(table: Type[T]) -> Sequence[T]:
    with Session(engine) as session:
        statement = select(table)
        result = session.exec(statement)
        return result.all()


def clear_table(table: Type[T]) -> None:
    with Session(engine) as db_session:
        db_session.exec(delete(table))
        db_session.commit()


def get_conflict_columns(engine, table):
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


def insert_if_not_exists(table: Type[T], instances: list[T]) -> None:
    if len(instances) == 0:
        return

    conflict_columns = get_conflict_columns(engine, table)

    def insert_one_batch(batch: list[T]) -> None:
        with Session(engine) as db_session:
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


def set_tournament_as_scraped(tournament: Tournament) -> None:
    with Session(engine) as session:
        attached_tournament = session.exec(
            select(Tournament).where(
                Tournament.tournament_id == tournament.tournament_id
            )
        ).first()
        if attached_tournament:
            attached_tournament.has_been_scraped = True
            session.commit()


def unset_all_tournament_as_scraped() -> None:
    with Session(engine) as session:
        tournaments = get_table(Tournament)
        for tournament in tournaments:
            tournament.has_been_scraped = False
            session.add(tournament)
        session.commit()


def get_player_by_name(player_name: str) -> Optional[Player]:
    with Session(engine) as session:
        statement = select(Player).where(Player.name.like(f"%{player_name}%"))
        result = session.exec(statement)
        return result.first()

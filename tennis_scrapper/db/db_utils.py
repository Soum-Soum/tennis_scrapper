from typing import Type, TypeVar

from sqlalchemy.dialects.sqlite import insert
from sqlmodel import SQLModel, Sequence, create_engine, Session, select, delete

from conf.config import settings
from db.models import Tournament

DB_PATH = settings.db_url

engine = create_engine(DB_PATH)
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


def insert_if_not_exists(table: Type[T], instances: list[T]) -> None:
    if len(instances) == 0:
        return
    with Session(engine) as db_session:
        values = [instance.model_dump() for instance in instances]
        stmt = insert(table).values(values).on_conflict_do_nothing()
        db_session.exec(stmt)
        db_session.commit()


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

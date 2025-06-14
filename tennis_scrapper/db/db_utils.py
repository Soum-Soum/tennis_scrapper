from typing import Type, TypeVar

from sqlalchemy.dialects.sqlite import insert
from sqlmodel import SQLModel, Sequence, create_engine, Session, select
from conf.config import settings

DB_PATH = settings.db_url

engine = create_engine(DB_PATH)
SQLModel.metadata.create_all(engine)


T = TypeVar("T", bound=SQLModel)


def get_table(table: Type[T]) -> Sequence[T]:
    with Session(engine) as session:
        statement = select(table)
        result = session.exec(statement)
        return result.all()


def insert_if_not_exists(table: Type[T], instances: list[T]) -> None:
    if len(instances) == 0:
        return
    with Session(engine) as db_session:
        values = [instance.model_dump() for instance in instances]
        stmt = insert(table).values(values).on_conflict_do_nothing()
        db_session.exec(stmt)
        db_session.commit()

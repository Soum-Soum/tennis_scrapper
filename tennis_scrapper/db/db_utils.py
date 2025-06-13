from typing import Type, TypeVar
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

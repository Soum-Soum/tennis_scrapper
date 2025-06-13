from sqlmodel import SQLModel, create_engine, Session
from conf.config import settings

DB_PATH = settings.db_url


def get_engine():
    return create_engine(DB_PATH)


def create_db_and_tables():
    engine = get_engine()
    SQLModel.metadata.create_all(engine)
    return engine


def get_session(engine=None):
    if engine is None:
        engine = get_engine()
    return Session(engine)

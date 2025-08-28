from typing import AsyncGenerator

from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine

from conf.config import settings


_ASYNC_DB_URL = getattr(settings, "async_db_url", None) or getattr(settings, "db_url")
_ASYNC_ENGINE = create_async_engine(_ASYNC_DB_URL, echo=False, future=True)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSession(_ASYNC_ENGINE, expire_on_commit=False) as session:
        yield session
